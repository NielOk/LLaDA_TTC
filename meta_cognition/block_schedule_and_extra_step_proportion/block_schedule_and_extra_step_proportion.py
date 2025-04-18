import torch
import numpy as np
import torch.nn.functional as F
import argparse

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


def compute_steps_per_block(block_schedule, total_steps, proportions):
    base_steps = block_schedule.copy()
    remaining_steps = total_steps - sum(base_steps)
    assert remaining_steps >= 0, "Not enough total steps to cover base allocation"
    assert len(block_schedule) == len(proportions), "Mismatch in block/proportion length"
    assert abs(sum(proportions) - 1.0) < 1e-5, "Proportions must sum to 1"

    extra_steps = [int(remaining_steps * p) for p in proportions]
    step_sum = sum(base + extra for base, extra in zip(base_steps, extra_steps))
    leftover = total_steps - step_sum
    for i in range(leftover):
        extra_steps[i % len(extra_steps)] += 1

    return [base + extra for base, extra in zip(base_steps, extra_steps)]


@torch.no_grad()
def generate(model, prompt, steps=160, gen_length=128, block_schedule=None, extra_step_proportions=None, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    if block_schedule is None:
        block_schedule = [gen_length]
        extra_step_proportions = [1.0]

    assert sum(block_schedule) == gen_length, "Block schedule must sum to total gen_length"
    steps_per_block = compute_steps_per_block(block_schedule, steps, extra_step_proportions)

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)

    block_start = prompt.shape[1]
    for block_id, (block_len, block_steps) in enumerate(zip(block_schedule, steps_per_block)):
        block_end = block_start + block_len
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, block_steps)

        for i in range(block_steps):
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
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, :block_start] = -np.inf
            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

        block_start += block_len

    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_variant', choices=['base', 'instruct'], default='instruct')
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

    block_schedule = [32, 48, 48]  # must sum to gen_length
    extra_step_proportions = [0.5, 0.3, 0.2]  # must sum to 1.0

    out = generate(model, input_ids, steps=160, gen_length=128, block_schedule=block_schedule,
                   extra_step_proportions=extra_step_proportions, temperature=0., cfg_scale=0., remasking='low_confidence')

    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()