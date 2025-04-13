import torch
import numpy as np
import torch.nn.functional as F
import argparse

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
@torch.no_grad()
def generate_with_dts(model, prompt, steps=128, gen_length=128, block_length=128,
                      temperature=0., cfg_scale=0., mask_id=126336,
                      search_width=4, branches_per_candidate=2,
                      remask_steps=3, alpha=0.7):
    """
    Denoising Trajectory Search (DTS): multi-path inference for LLaDA-style diffusion LMs.
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        mask_id: The token id of [MASK] is 126336.
        search_width: The number of candidates to keep at each step.
        branches_per_candidate: The number of branches to explore for each candidate.
        remask_steps: The number of remask-denoise iterations.
        alpha: The weight for entropy in the remasking score.
    """
    prompt_len = prompt.shape[1]
    total_len = prompt_len + gen_length
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    base_x = torch.full((1, total_len), mask_id, dtype=torch.long).to(model.device)
    base_x[:, :prompt_len] = prompt
    beam = [(base_x, torch.tensor(float('inf')).to(model.device))]  # (sequence, score)

    for num_block in range(num_blocks):
        new_beam = []

        for x, _ in beam:
            for _ in range(branches_per_candidate):
                x_branch = x.clone()

                for _ in range(remask_steps):
                    # CFG forward pass (initial prediction)
                    if cfg_scale > 0.:
                        un_x = x_branch.clone()
                        un_x[:, :prompt_len] = mask_id
                        x_ = torch.cat([x_branch, un_x], dim=0)
                        logits = model(x_).logits
                        logits, un_logits = torch.chunk(logits, 2, dim=0)
                        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                    else:
                        logits = model(x_branch).logits

                    logits = logits.to(torch.float64)
                    probs = F.softmax(logits, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                    rand = torch.rand_like(entropy)
                    remask_score = alpha * entropy + (1 - alpha) * rand

                    # Restrict scoring to current block
                    block_start = prompt_len + num_block * block_length
                    block_end = prompt_len + (num_block + 1) * block_length
                    remask_score[:, :block_start] = -float('inf')
                    remask_score[:, block_end:] = -float('inf')

                    # Select top-k tokens to remask
                    k = block_length // steps_per_block
                    remask_index = torch.zeros_like(x_branch, dtype=torch.bool)
                    for b in range(x_branch.shape[0]):
                        _, idx = torch.topk(remask_score[b], k)
                        remask_index[b, idx] = True

                    # Remask them
                    x_branch[remask_index] = mask_id

                    # Forward pass again after remasking
                    if cfg_scale > 0.:
                        un_x = x_branch.clone()
                        un_x[:, :prompt_len] = mask_id
                        x_ = torch.cat([x_branch, un_x], dim=0)
                        logits = model(x_).logits
                        logits, un_logits = torch.chunk(logits, 2, dim=0)
                        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                    else:
                        logits = model(x_branch).logits

                    logits = logits.to(torch.float64)
                    gumbel_noise = -torch.empty_like(logits).exponential_().log()
                    logits_with_noise = logits + gumbel_noise * temperature
                    x0 = torch.argmax(logits_with_noise, dim=-1)

                    # Apply new predictions
                    x_branch[remask_index] = x0[remask_index]

                # Score full sequence (mean entropy over masked positions)
                mask = (x_branch == mask_id)
                logits = model(x_branch).logits
                probs = F.softmax(logits, dim=-1)
                token_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                score = token_entropy[mask].mean() if mask.any() else token_entropy.mean()

                new_beam.append((x_branch, score))

        # Keep best `search_width` candidates
        new_beam.sort(key=lambda tup: tup[1].item())
        beam = new_beam[:search_width]

    return beam[0][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_variant', choices=['base', 'instruct'], default='instruct',
                        help="Choose 'base' or 'instruct' model variant.")
    args = parser.parse_args()

    device = 'cuda'

    if args.model_variant == 'instruct':
        model_name = 'GSAI-ML/LLaDA-8B-Instruct'
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

        prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt)['input_ids']
    else:
        model_name = 'GSAI-ML/LLaDA-8B-Base'
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

        prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
        input_ids = tokenizer(prompt)['input_ids']

    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate_with_dts(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0.,
                   cfg_scale=0., remasking='low_confidence')

    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])
    

if __name__ == '__main__':
    main()