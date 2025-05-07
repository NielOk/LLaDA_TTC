""" GRPO-embedded Monte Carlo Tree Search over Decoding Policies """

import torch
import numpy as np
import torch.nn.functional as F
import argparse
import random

from transformers import AutoTokenizer, AutoModel
from decoding_policy_state import DecodingPolicyState
from policy_based_decoding_utils import *
from mcts_node import MCTSNode
from grpo_embedded_mcts_utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_variant', choices=['base', 'instruct'], default='instruct',
                        help="Choose 'base' or 'instruct' model variant.")
    args = parser.parse_args()

    device = 'cuda'

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

    # Initialize the model and tokenizer
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
    out = generate_with_decoding_policy(model, input_ids, decoding_policy_state, steps=steps, gen_length=gen_length, cfg_scale=0., mask_id=126336)
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()