""" Monte Carlo Tree Search over Decoding Policies """

import torch
import numpy as np
import torch.nn.functional as F
import argparse
import random

from transformers import AutoTokenizer, AutoModel
from decoding_policy_state import DecodingPolicyState

@ torch.no_grad()
def generate_with_decoding_policy(model, prompt, decoding_policy, steps=128, gen_length=128, block_length=128, cfg_scale=0., mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        decoding_policy: Decoding policy state object.
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        cfg_scale: Unsupervised classifier-free guidance scale.
        mask_id: The token id of [MASK] is 126336.
    '''
    pass


def main():
    cur_decoding_policy_state = None
    possible_temperatures = [i / 10 for i in range(0, 11)]
    possible_remasking_strategies= ["low_confidence", "random"]
    steps = 256
    gen_length = 128
    max_num_blocks = 4
    decoding_policy_state = DecodingPolicyState()
    for i in range(steps):
        decoding_policy_state.sample_partial_decoding_policy(possible_temperatures=possible_temperatures, 
                                                                   possible_remasking_strategies=possible_remasking_strategies, 
                                                                   steps=steps, 
                                                                   gen_length=gen_length, 
                                                                   max_num_blocks=max_num_blocks)
        
    print("Decoding policy state temperature schedule: ", decoding_policy_state.temperature_schedule)
    print("Decoding policy state remasking strategy schedule: ", decoding_policy_state.remasking_strategy_schedule)
    print("Decoding policy state block schedule: ", decoding_policy_state.block_schedule)
    print("Decoding policy state extra step proportions: ", decoding_policy_state.extra_step_proportions)
    print("Decoding policy state step id: ", decoding_policy_state.step_id)
    print("Decoding policy state block id: ", decoding_policy_state.block_id)
    print("Decoding policy state block end step id: ", decoding_policy_state.block_end_step_id)

if __name__ == '__main__':
    main()