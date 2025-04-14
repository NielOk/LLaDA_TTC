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


@torch.no_grad()
def compute_entropy_influence(model, x_branch, mask_id, block_start, block_end):
    """
    Computes influence score for each token by measuring entropy increase when that token is masked.
    Influence is L2-normalized across each sequence.
    """
    base_logits = model(x_branch).logits  # [B, T, V]
    base_probs = F.softmax(base_logits, dim=-1)
    base_entropy = -torch.sum(base_probs * base_probs.log(), dim=-1)  # [B, T]

    influence = torch.zeros_like(base_entropy)

    for j in range(block_start, block_end):
        masked = x_branch.clone()
        masked[:, j] = mask_id
        masked_logits = model(masked).logits
        masked_probs = F.softmax(masked_logits, dim=-1)
        masked_entropy = -torch.sum(masked_probs * masked_probs.log(), dim=-1)

        delta = masked_entropy - base_entropy  # [B, T]
        influence += delta

    influence = influence / (block_end - block_start)

    # Zero out outside block
    influence[:, :block_start] = 0
    influence[:, block_end:] = 0

    # L2 normalization per sequence
    norm = torch.norm(influence, p=2, dim=1, keepdim=True) + 1e-8
    influence = influence / norm

    return influence


@torch.no_grad()
def generate_with_dts_token_dependency_integration(model, prompt, steps=128, gen_length=128, block_length=128,
                      temperature=0., cfg_scale=0., mask_id=126336,
                      search_width=4, branches_per_candidate=2,
                      remask_steps=3,
                      entropy_weight=0.7, influence_weight=0.3, stochasticity_weight=0.0):
    """
    Denoising Trajectory Search (DTS) with tunable remask score components.
    """
    total_weight = entropy_weight + influence_weight + stochasticity_weight
    assert abs(total_weight - 1.0) < 1e-6, "entropy + influence + stochasticity weights must sum to 1.0"

    prompt_len = prompt.shape[1]
    total_len = prompt_len + gen_length
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    base_x = torch.full((1, total_len), mask_id, dtype=torch.long).to(model.device)
    base_x[:, :prompt_len] = prompt
    beam = [(base_x, torch.tensor(float('inf')).to(model.device))]

    for num_block in range(num_blocks):
        new_beam = []

        for x, _ in beam:
            for _ in range(branches_per_candidate):
                x_branch = x.clone()

                for _ in range(remask_steps):
                    # CFG forward pass
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

                    block_start = prompt_len + num_block * block_length
                    block_end = prompt_len + (num_block + 1) * block_length
                    influence = compute_entropy_influence(model, x_branch, mask_id, block_start, block_end)
                    noise = torch.rand_like(entropy)

                    # Final remask score
                    remask_score = (
                        entropy_weight * entropy +
                        influence_weight * influence +
                        stochasticity_weight * noise
                    )

                    remask_score[:, :block_start] = -float('inf')
                    remask_score[:, block_end:] = -float('inf')

                    k = block_length // steps_per_block
                    remask_index = torch.zeros_like(x_branch, dtype=torch.bool)
                    for b in range(x_branch.shape[0]):
                        _, idx = torch.topk(remask_score[b], k)
                        remask_index[b, idx] = True

                    x_branch[remask_index] = mask_id

                    # Redenoise
                    if cfg_scale > 0.:
                        un_x = x_branch.clone()
                        un_x[:, :prompt_len] = mask_id
                        x_ = torch.cat([x_branch, un_x], dim=0)
                        logits = model(x_).logits
                        logits, un_logits = torch.chunk(logits, 2, dim=0)
                        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                    else:
                        logits = model(x_branch).logits

                    gumbel_logits = add_gumbel_noise(logits, temperature)
                    x0 = torch.argmax(gumbel_logits, dim=-1)
                    x_branch[remask_index] = x0[remask_index]

                # Score final
                mask = (x_branch == mask_id)
                logits = model(x_branch).logits
                probs = F.softmax(logits, dim=-1)
                token_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                score = token_entropy[mask].mean() if mask.any() else token_entropy.mean()

                new_beam.append((x_branch, score))

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

    steps=16
    gen_length=1024
    block_length=64
    temperature=0.
    cfg_scale=0.
    mask_id=126336
    search_width=4
    branches_per_candidate=2
    remask_steps=3,
    entropy_weight=0.6
    influence_weight=0.3
    stochasticity_weight=0.1
    out = generate_with_dts_token_dependency_integration(model, input_ids, steps=steps, gen_length=gen_length, block_length=block_length, temperature=temperature,
                        cfg_scale=cfg_scale, mask_id=mask_id,
                        search_width=search_width, branches_per_candidate=branches_per_candidate,
                        remask_steps=remask_steps,
                        entropy_weight=entropy_weight, influence_weight=influence_weight, stochasticity_weight=stochasticity_weight)

    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])
    

if __name__ == '__main__':
    main()