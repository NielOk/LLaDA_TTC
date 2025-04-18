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
        }
        cur_step = 0
        block_id = 0
    else:
        block_id = cur_decoding_policy_state["block_id"]
        cur_step = cur_decoding_policy_state["step_id"]

    # Sample a temperature
    if cur_step < steps:
        temperature = random.choice(possible_temperatures)
        cur_decoding_policy_state["temperature_schedule"].append(temperature)
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