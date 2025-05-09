""" GRPO-embedded Monte Carlo Tree Search over Decoding Policies """

import torch
import numpy as np
import torch.nn.functional as F
import argparse
import random
from datasets import load_dataset
import json

from transformers import AutoTokenizer, AutoModel
from decoding_policy_state import DecodingPolicyState
from policy_based_decoding_utils import *
from mcts_node import MCTSNode
from grpo_embedded_mcts_utils import *

def load_folio_train_dataset(num_questions_to_sample=1000):
    ds = load_dataset("yale-nlp/FOLIO", split="train")
    
    to_sample = random.sample(list(ds), num_questions_to_sample)
    formatted_questions = []
    labels = []

    for sample in to_sample:
        premise = sample["premises"]
        conclusion = sample["conclusion"]
        label = sample["label"]

        formatted_question = f'The following facts are given: {premise}. Is the conclusion "{conclusion}" logically entailed by the above facts? Think out loud step by step and, on a new line, write one of the following words by itself as your answer: "False", "Uncertain", or "True".'
        formatted_questions.append((formatted_question))
        labels.append(label)
    
    return formatted_questions, labels

def main():
    device = 'cuda'
    
    num_folio_questions_to_sample = 5 # Trying smaller set for testing

    model_name = 'GSAI-ML/LLaDA-8B-Instruct' # Use instruct model by default

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

    train_formatted_questions, train_labels = load_folio_train_dataset(num_questions_to_sample=num_folio_questions_to_sample)
    input_ids_list = []
    for prompt in train_formatted_questions:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
        input_ids_list.append(input_ids)

    top_policies = search_shared(
        model=model,
        prompts=input_ids_list,
        labels=train_labels,
        steps=128,
        iters=50,
        branching_factor=2,
        top_k=3,
        possible_temperatures=[0.7, 1.0],
        possible_remasking_strategies=["low_confidence", "random"],
        gen_length=128,
        max_num_blocks=4
        )
    
    # Save the top policies
    top_policies_dict = {}
    top_policies_dict['metadata'] = {
        "description": "Top policies from GRPO-embedded MCTS over decoding policies",
        "model_name": model_name,
        "num_questions_sampled": num_folio_questions_to_sample,
        "folio_dataset_examples": train_formatted_questions,
        "folio_dataset_labels": train_labels,
    }
    for i, policy in enumerate(top_policies):
        top_policies_dict[f"policy_{i}"] = {
            "temperature_schedule": policy.temperature_schedule,
            "remasking_strategy_schedule": policy.remasking_strategy_schedule,
            "block_schedule": policy.block_schedule,
            "extra_step_proportions": policy.extra_step_proportions
        }


if __name__ == '__main__':
    main()