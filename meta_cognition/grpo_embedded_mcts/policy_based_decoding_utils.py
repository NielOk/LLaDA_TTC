import torch
import numpy as np
import torch.nn.functional as F


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
def generate_with_decoding_policy(model, prompt, decoding_policy, steps=128, gen_length=128, cfg_scale=0., mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        decoding_policy: DecodingPolicyState object (fully sampled).
        steps: Total number of sampling steps.
        gen_length: Number of tokens to generate.
        cfg_scale: Classifier-free guidance scale.
        mask_id: Token ID for [MASK].
    '''
    
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)

    assert sum(decoding_policy.block_schedule) == gen_length, "Block schedule must sum to total gen_length."
    assert len(decoding_policy.temperature_schedule) == len(decoding_policy.block_schedule)
    assert len(decoding_policy.remasking_strategy_schedule) == len(decoding_policy.block_schedule)
    assert len(decoding_policy.extra_step_proportions) == len(decoding_policy.block_schedule)

    # Convert proportions to actual steps
    raw_steps = [p * steps for p in decoding_policy.extra_step_proportions]
    int_steps = [int(s) for s in raw_steps]
    # Distribute any leftover steps due to rounding
    while sum(int_steps) < steps:
        diffs = [r - i for r, i in zip(raw_steps, int_steps)]
        idx = diffs.index(max(diffs))
        int_steps[idx] += 1

    step_ptr = 0
    token_ptr = prompt.shape[1]

    for block_id, block_len in enumerate(decoding_policy.block_schedule):
        block_steps = int_steps[block_id]
        block_start = token_ptr
        block_end = token_ptr + block_len

        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, block_steps)

        temperature = decoding_policy.temperature_schedule[block_id]
        remasking = decoding_policy.remasking_strategy_schedule[block_id]

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
                raise NotImplementedError(f"Unknown remasking strategy: {remasking}")

            # Only allow remasking within current block
            x0_p[:, :block_start] = -np.inf
            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True

            x[transfer_index] = x0[transfer_index]

        step_ptr += block_steps
        token_ptr += block_len

    return x