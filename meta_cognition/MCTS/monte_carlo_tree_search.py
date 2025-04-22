""" Monte Carlo Tree Search over Decoding Policies """

import torch
import numpy as np
import torch.nn.functional as F
import argparse
import random

from transformers import AutoTokenizer, AutoModel


def sample_partial_decoding_policy(cur_decoding_policy_state=None, possible_temperatures=[], possible_remasking_strategies=["low_confidence", "random"], steps=128, gen_length=128, max_num_blocks=4):
    """
    Sample a partial decoding policy from the space of possible remaining 
    decoding policies. Each partial decoding policy should output a state dict of the
    decoding policy, which can be used to sample the next decoding step.
    The dictionary should be of form:
        "cur_decoding_policy_state": {
            "temperature_schedule": [list of temperatures to this step],
            "remasking_strategy_schedule": [list of remasking schedules to this step],
            "block_schedule": [list of block lengths to this step],
            "extra_step_proportions": [list of extra step proportions to this step. length should be equal to the number of blocks],
            "step_id": id of the current step,
            "block_id": id of the current block,
        }
    """

    if cur_decoding_policy_state is None:
        cur_decoding_policy_state = {
            "temperature_schedule": [],
            "remasking_strategy_schedule": [],
            "block_schedule": [],
            "extra_step_proportions": [],
            "step_id": 0,
            "block_id": 0,
            "block_end_step_id": 0
        }
        cur_step = 0
        block_id = 0
        block_end_step_id = 0
    else:
        block_id = cur_decoding_policy_state["block_id"]
        cur_step = cur_decoding_policy_state["step_id"]
        block_end_step_id = cur_decoding_policy_state["block_end_step_id"]

    # Sample a new partial policy state
    if cur_step < steps:
        # Sample a temperature from the possible temperatures
        temperature = random.choice(possible_temperatures)
        cur_decoding_policy_state["temperature_schedule"].append(temperature)

        # Sample a remasking strategy
        remasking_strategy = random.choice(possible_remasking_strategies)
        cur_decoding_policy_state["remasking_strategy_schedule"].append(remasking_strategy)

        # Sample a block length and extra step proportion if eligible
        if cur_step == block_end_step_id and block_id < max_num_blocks:
            blocks_remaining = max_num_blocks - block_id

            # Sample a block length from the possible block lengths
            if blocks_remaining == 1: # Scenario where we only have 1 block remaining when time to sample a new block
                block_length = gen_length - cur_step
                extra_step_proportion = round(1 - sum(cur_decoding_policy_state["extra_step_proportions"]), 2)
            else: # Scenario where we have multiple blocks remaining
                block_length = random.randint(1, gen_length - cur_step - (blocks_remaining - 1))
                extra_step_proportion = round(random.uniform(0, 1 - sum(cur_decoding_policy_state["extra_step_proportions"])), 2)

            # Update decoding policy state
            cur_decoding_policy_state["block_end_step_id"] = cur_step + block_length
            cur_decoding_policy_state["block_schedule"].append(block_length)
            cur_decoding_policy_state["extra_step_proportions"].append(extra_step_proportion)
            cur_decoding_policy_state["block_id"] += 1

    else:
        print("All steps have been sampled. No more partial decoding policies can be sampled.")
        return cur_decoding_policy_state

    cur_decoding_policy_state["step_id"] += 1

    return cur_decoding_policy_state


def main():
    cur_decoding_policy_state = None
    possible_temperatures = [i / 10 for i in range(0, 11)]
    possible_remasking_strategies= ["low_confidence", "random"]
    steps = 128
    gen_length = 128
    max_num_blocks = 4
    for i in range(steps):
        cur_decoding_policy_state = sample_partial_decoding_policy(cur_decoding_policy_state=cur_decoding_policy_state,
                                                                   possible_temperatures=possible_temperatures, 
                                                                   possible_remasking_strategies=possible_remasking_strategies, 
                                                                   steps=steps, 
                                                                   gen_length=gen_length, 
                                                                   max_num_blocks=max_num_blocks)
        

if __name__ == '__main__':
    main()