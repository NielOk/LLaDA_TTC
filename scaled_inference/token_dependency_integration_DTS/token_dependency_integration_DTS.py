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


def compute_influence_scores(model, x_branch, logits, mask_id, block_start, block_end):
    with torch.no_grad():
        influence = torch.zeros_like(logits[..., 0])
        base_probs = F.softmax(logits, dim=-1)
        for j in range(block_start, block_end):
            masked_input = x_branch.clone()
            masked_input[:, j] = mask_id
            masked_logits = model(masked_input).logits
            masked_probs = F.softmax(masked_logits, dim=-1)
            kl = F.kl_div(masked_probs.log(), base_probs, reduction='none').sum(dim=-1)
            influence += kl
        influence = influence / (block_end - block_start)
        influence[:, :block_start] = 0
        influence[:, block_end:] = 0
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
                    influence = compute_influence_scores(model, x_branch, logits, mask_id, block_start, block_end)
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