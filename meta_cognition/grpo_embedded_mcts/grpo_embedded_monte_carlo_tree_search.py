""" GRPO-embedded Monte Carlo Tree Search over Decoding Policies """

import torch
import numpy as np
import torch.nn.functional as F
import argparse
import random
from datasets import load_dataset
import json
from huggingface_hub import login
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoModel
from decoding_policy_state import DecodingPolicyState
from policy_based_decoding_utils import *
from mcts_node import MCTSNode
from grpo_embedded_mcts_utils import *

load_dotenv()
# Load huggingface token
hugging_face_token = os.getenv("niel_hugging_face_token")
login(token=hugging_face_token)

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

def convert_prompts_to_input_ids(prompts, tokenizer, device):
    input_ids_list = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
        input_ids_list.append(input_ids)

    return input_ids_list

def train_tree_from_scratch(device, tokenizer, model, steps, iters, branching_factor, top_k, model_name, metadata_filename, tree_filename,**sampling_kwargs):
    """
    Perform GRPO-embedded MCTS over decoding polciies from scratch"""

    num_folio_questions_to_sample_train = 5 # Trying small set for training
    num_folio_questions_to_sample_test = 1 # Try smaller set for testing

    # Get train and test sets
    all_formatted_questions, all_labels = load_folio_train_dataset(num_questions_to_sample=num_folio_questions_to_sample_train + num_folio_questions_to_sample_test)
    train_formatted_questions, train_labels = all_formatted_questions[:num_folio_questions_to_sample_train], all_labels[:num_folio_questions_to_sample_train]
    test_formatted_questions, test_labels = all_formatted_questions[num_folio_questions_to_sample_train:], all_labels[num_folio_questions_to_sample_train:]
    
    # Convert training set to input ids
    train_input_ids_list = convert_prompts_to_input_ids(train_formatted_questions, tokenizer, device)

    root, top_policies = search_shared(
        model=model, 
        tokenizer=tokenizer,
        prompts=train_input_ids_list,
        labels=train_labels,
        steps=steps,
        iters=iters,
        branching_factor=branching_factor,
        top_k=top_k,
        **sampling_kwargs,
    )

    # Save the top policies and metadata
    top_policies_dict = {}
    top_policies_dict['metadata'] = {
        "description": "Top policies from GRPO-embedded MCTS over decoding policies",
        "model_name": model_name,
        "num_questions_sampled_for_training": num_folio_questions_to_sample_train,
        "num_questions_sampled_for_testing": num_folio_questions_to_sample_test,
        "num_questions_sampled": num_folio_questions_to_sample_train + num_folio_questions_to_sample_test,
        "folio_train_dataset_prompts": train_formatted_questions,
        "folio_train_dataset_labels": train_labels,
        "folio_test_dataset_prompts": test_formatted_questions,
        "folio_test_dataset_labels": test_labels,
        "steps": steps,
        "iters": iters,
        "branching_factor": branching_factor,
        "top_k": top_k,
        "possible_temperatures": sampling_kwargs["possible_temperatures"],
        "possible_remasking_strategies": sampling_kwargs["possible_remasking_strategies"],
        "gen_length": sampling_kwargs["gen_length"],
        "max_num_blocks": sampling_kwargs["max_num_blocks"]
    }
    for i, policy in enumerate(top_policies):
        top_policies_dict[f"policy_{i}"] = {
            "temperature_schedule": policy.temperature_schedule,
            "remasking_strategy_schedule": policy.remasking_strategy_schedule,
            "block_schedule": policy.block_schedule,
            "extra_step_proportions": policy.extra_step_proportions
        }

    # Save the tree
    with open("mcts_tree_snapshot.json", "w") as f:
        json.dump(root.to_dict(), f, indent=2)

def main():
    device = 'cuda'

    print("started model loading...")

    model_name = 'GSAI-ML/LLaDA-8B-Instruct' # Use instruct model by default

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

    print("Model loaded successfully.")

    steps = 256
    iters = 1
    branching_factor = 2 # number of children to sample at each node
    top_k = 3
    possible_temperatures = [0.7, 1.0]
    possible_remasking_strategies = ["low_confidence", "random"]
    gen_length = 128
    max_num_blocks = 4 # Depth of the tree

    sampling_kwargs = {
        "possible_temperatures": possible_temperatures,
        "possible_remasking_strategies": possible_remasking_strategies,
        "gen_length": gen_length,
        "max_num_blocks": max_num_blocks,
    }

    # Train the tree from scratch
    metadata_filename = "mcts_metadata.json"
    tree_filename = "mcts_tree_snapshot.json"
    train_tree_from_scratch(device, tokenizer, model, steps, iters, branching_factor, top_k, model_name, metadata_filename, tree_filename, **sampling_kwargs)

    # Load the tree
    with open(tree_filename) as f:
        root = MCTSNode.from_dict(json.load(f))

if __name__ == '__main__':
    main()