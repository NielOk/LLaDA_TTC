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
def calculate_pre_generation_entropy(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    initial_logits = model(x).logits  # (batch, seq_len, vocab_size)
    probs = F.softmax(initial_logits.to(torch.float64), dim=-1)  # high precision

    # Compute entropy over the vocabulary for each token
    token_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)  # (batch, seq_len)

    # Only consider masked positions
    mask_index = (x == mask_id)
    masked_token_entropy = token_entropy[mask_index]  # 1D tensor

    # Compute statistics
    mean_entropy = masked_token_entropy.mean()
    max_entropy = masked_token_entropy.max()

    return mean_entropy, max_entropy


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


@torch.no_grad()
def calculate_post_generation_entropy(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Calculates mean and max entropy *after* generation, using the same masked positions
    as in pre-generation entropy for a fair comparison.
    '''
    # Generate the output
    output = generate(model, prompt, steps, gen_length, block_length, temperature, cfg_scale, remasking, mask_id)

    # Mask identifying generated (non-prompt) positions
    gen_mask = torch.zeros_like(output, dtype=torch.bool)
    gen_mask[:, prompt.shape[1]:] = True

    # Feed the generated sequence back into the model
    final_logits = model(output).logits  # (batch, seq_len, vocab_size)
    probs = F.softmax(final_logits.to(torch.float64), dim=-1)  # high precision

    # Compute entropy at each position
    token_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)  # (batch, seq_len)

    # Filter to only generated tokens
    gen_token_entropy = token_entropy[gen_mask]  # 1D tensor

    # Compute statistics
    mean_entropy = gen_token_entropy.mean()
    max_entropy = gen_token_entropy.max()

    return mean_entropy, max_entropy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_variant', choices=['base', 'instruct'], default='instruct',
                        help="Choose 'base' or 'instruct' model variant.")
    args = parser.parse_args()

    device = 'cuda'

    # Prompts to evaluate entropy of model inference start for
    prompts = [
            "What is the capital of France?",
            "Explain the theory of relativity.",
            "How does photosynthesis work?",
            "What are the benefits of exercise?",
            "Describe the process of cellular respiration.",
            "Explain the concept of quantum mechanics.",
            "Explain the concept of gravity."
        ]

    if args.model_variant == 'instruct':
        model_name = 'GSAI-ML/LLaDA-8B-Instruct'
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

        input_ids_list = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            input_ids = tokenizer(prompt)['input_ids']
            input_ids_list.append(input_ids)
    else:
        model_name = 'GSAI-ML/LLaDA-8B-Base'
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
        
        input_ids_list = []
        for prompt in prompts:
            input_ids = tokenizer(prompt)['input_ids']
            input_ids_list.append(input_ids)

    # Calculate entropys
    for i in range(len(input_ids_list)):
        input_ids = input_ids_list[i]
        prompt = prompts[i]
        input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
        print(f"Prompt: {prompt}")

        # Pre-generation entropy
        mean_pre_generation_entropy, max_pre_generation_entropy = calculate_pre_generation_entropy(model, input_ids)
        print(f"Mean Pre-Generation Entropy: {mean_pre_generation_entropy.item():.4f}, Max Pre-Generation Entropy: {max_pre_generation_entropy.item():.4f}")

        mean_post_generation_entropy, max_post_generation_entropy = calculate_post_generation_entropy(model, input_ids)
        print(f"Mean Post-Generation Entropy: {mean_post_generation_entropy.item():.4f}, Max Post-Generation Entropy: {max_post_generation_entropy.item():.4f}")


if __name__ == '__main__':
    main()