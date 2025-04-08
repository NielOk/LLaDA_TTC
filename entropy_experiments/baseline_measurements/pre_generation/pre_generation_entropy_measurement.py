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
        mean_entropy, max_entropy = calculate_pre_generation_entropy(model, input_ids)
        print(f"Prompt {prompt}: Mean Entropy: {mean_entropy.item():.4f}, Max Entropy: {max_entropy.item():.4f}")


if __name__ == '__main__':
    main()