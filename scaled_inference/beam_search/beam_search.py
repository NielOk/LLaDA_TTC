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


@torch.no_grad()
def generate_with_beam_denoising(model, prompt, steps=64, gen_length=128, block_length=32,
                                  beam_width=4, branches_per_candidate=2, remask_steps=2,
                                  temperature=0.7, alpha=0.7, mask_id=None):
    device = model.device
    prompt_len = prompt.shape[1]
    total_len = prompt_len + gen_length
    num_blocks = gen_length // block_length
    steps_per_block = max(1, steps // num_blocks)

    base_x = torch.full((1, total_len), mask_id, dtype=torch.long, device=device)
    base_x[:, :prompt_len] = prompt
    beam = [(base_x, torch.tensor(float('inf'), device=device))]


    for block_idx in range(num_blocks):
        new_beam = []
        for x, _ in beam:
            for _ in range(branches_per_candidate):
                x_branch = x.clone()
                for _ in range(remask_steps):
                    logits = model(x_branch).logits
                    probs = F.softmax(logits, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                    rand = torch.rand_like(entropy)
                    remask_score = alpha * entropy + (1 - alpha) * rand

                    block_start = prompt_len + block_idx * block_length
                    block_end = prompt_len + (block_idx + 1) * block_length
                    remask_score[:, :block_start] = -float('inf')
                    remask_score[:, block_end:] = -float('inf')

                    k = max(1, block_length // steps_per_block)
                    remask_index = torch.zeros_like(x_branch, dtype=torch.bool)
                    for b in range(x_branch.shape[0]):
                        _, idx = torch.topk(remask_score[b], k)
                        remask_index[b, idx] = True

                    x_branch[remask_index] = mask_id

                    logits = model(x_branch).logits
                    gumbel_logits = add_gumbel_noise(logits, temperature)
                    x0 = torch.argmax(gumbel_logits, dim=-1)
                    x_branch[remask_index] = x0[remask_index]

                # Score: mean entropy on gen region
                logits = model(x_branch).logits
                probs = F.softmax(logits, dim=-1)
                token_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                gen_entropy = token_entropy[:, prompt_len:]
                score = gen_entropy.mean()
                new_beam.append((x_branch, score))

        new_beam.sort(key=lambda tup: tup[1].item())
        beam = new_beam[:beam_width]

    return beam[0][0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_variant', choices=['base', 'instruct'], default='instruct')
    args = parser.parse_args()

    device = 'cuda'

    if args.model_variant == 'instruct':
        model_name = 'GSAI-ML/LLaDA-8B-Instruct'
    else:
        model_name = 'GSAI-ML/LLaDA-8B-Base'

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
        model.resize_token_embeddings(len(tokenizer))
    mask_id = tokenizer.mask_token_id

    prompt_text = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    if args.model_variant == 'instruct':
        messages = [{"role": "user", "content": prompt_text}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    else:
        input_ids = tokenizer(prompt_text, return_tensors='pt').input_ids.to(device)

    out = generate_with_beam_denoising(
        model, input_ids,
        steps=128,
        gen_length=128,
        block_length=32,
        beam_width=4,
        branches_per_candidate=2,
        remask_steps=2,
        temperature=0.7,
        alpha=0.7,
        mask_id=mask_id
    )

    print(tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True))

if __name__ == '__main__':
    main()
