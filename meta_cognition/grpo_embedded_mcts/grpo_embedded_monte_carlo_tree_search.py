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
        "description": "Top policies and metadata from GRPO-embedded MCTS over decoding policies pretraining",
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
        top_policies_dict[f"pretrained_policy_{i}"] = {
            "temperature_schedule": policy.temperature_schedule,
            "remasking_strategy_schedule": policy.remasking_strategy_schedule,
            "block_schedule": policy.block_schedule,
            "extra_step_proportions": policy.extra_step_proportions
        }

    # Save the metadata
    with open(metadata_filename, "w") as f:
        json.dump(top_policies_dict, f, indent=2)
    print(f"Saved metadata to {metadata_filename}")

    # Save the tree
    with open("mcts_tree_snapshot.json", "w") as f:
        json.dump(root.to_dict(), f, indent=2)
    print(f"Saved tree to {tree_filename}")

def train_additional_iters(num_additional_iters, device, tokenizer, model, metadata_filename, tree_filename):
    """
    Load the tree and metadata, and perform additional iterations of GRPO-embedded MCTS.
    """
    with open(metadata_filename) as f:
        metadata = json.load(f)
    with open(tree_filename) as f:
        root = MCTSNode.from_dict(json.load(f))

    steps = metadata['metadata']['steps']
    branching_factor = metadata['metadata']['branching_factor']
    top_k = metadata['metadata']['top_k']

    sampling_kwargs = {
        "possible_temperatures": metadata['metadata']['possible_temperatures'],
        "possible_remasking_strategies": metadata['metadata']['possible_remasking_strategies'],
        "gen_length": metadata['metadata']['gen_length'],
        "max_num_blocks": metadata['metadata']['max_num_blocks']
    }

    train_formatted_questions = metadata['metadata']['folio_train_dataset_prompts']
    train_labels = metadata['metadata']['folio_train_dataset_labels']

    train_input_ids_list = convert_prompts_to_input_ids(train_formatted_questions, tokenizer, device)

    # Perform additional iterations of GRPO-embedded MCTS
    resume_node = root
    root, top_policies = search_shared(
        model=model, 
        tokenizer=tokenizer,
        prompts=train_input_ids_list,
        labels=train_labels,
        steps=steps,
        iters=num_additional_iters,
        branching_factor=branching_factor,
        top_k=top_k,
        resume_node=resume_node,
        **sampling_kwargs,
    )

    # Save the top policies and metadata, only update is the number of iterations
    metadata['metadata']['iters'] += num_additional_iters
    with open(metadata_filename, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_filename}")
    
    # Save the updated tree
    with open(tree_filename, "w") as f:
        json.dump(root.to_dict(), f, indent=2)
    print(f"Saved updated metadata to {metadata_filename}")

def evaluate_policy(model, tokenizer, prompt, label, policy, steps, **sampling_kwargs):
    '''
    Evaluate a single policy on a single prompt and label and score it.
    '''
    output = generate_with_decoding_policy(model, prompt, policy, steps=steps, gen_length=sampling_kwargs['gen_length'])
    decoded_output = tokenizer.batch_decode(output[:, prompt.shape[1]:], skip_special_tokens=True)[0]

    print(f"Prompt: {prompt}")
    print(f"Decoded output: {decoded_output}")
    print(f"Label: {label}")
    reward = compute_reward(decoded_output, label)
    print(f"Reward: {reward}")

    return decoded_output, reward

def test_time_grpo_embedded_mcts(test_time_iters, device, tokenizer, model, pre_trained_metadata_filename, pre_trained_tree_filename, test_time_metadata_filename, test_time_tree_filename):
    '''
    Load the tree and metadata, then on the test set, perform GRPO-embedded MCTS, collect top policies, and score
    '''
    with open(pre_trained_metadata_filename) as f:
        metadata = json.load(f)
    with open(pre_trained_tree_filename) as f:
        root = MCTSNode.from_dict(json.load(f))

    steps = metadata['metadata']['steps']
    branching_factor = metadata['metadata']['branching_factor']
    top_k = metadata['metadata']['top_k']

    sampling_kwargs = {
        "possible_temperatures": metadata['metadata']['possible_temperatures'],
        "possible_remasking_strategies": metadata['metadata']['possible_remasking_strategies'],
        "gen_length": metadata['metadata']['gen_length'],
        "max_num_blocks": metadata['metadata']['max_num_blocks']
    }

    test_formatted_questions = metadata['metadata']['folio_test_dataset_prompts']
    test_labels = metadata['metadata']['folio_test_dataset_labels']

    test_input_ids_list = convert_prompts_to_input_ids(test_formatted_questions, tokenizer, device)

    # Perform test-time GRPO-embedded MCTS on the pre-trained tree
    resume_node = root
    test_time_root, test_time_top_policies = search_shared(
        model=model, 
        tokenizer=tokenizer,
        prompts=test_input_ids_list,
        labels=test_labels,
        steps=steps,
        iters=test_time_iters,
        branching_factor=branching_factor,
        top_k=top_k,
        resume_node=resume_node,
        **sampling_kwargs,
    )

    # Save and evaluate the top policies and metadata 
    metadata['metadata']['description'] = 'Top policies and metadata from GRPO-embedded MCTS over decoding policies at test time'
    metadata['metadata']['test_time_iters'] = test_time_iters

    for i, policy in enumerate(test_time_top_policies):
        metadata[f"test_time_policy_{i}"] = {
            "temperature_schedule": policy.temperature_schedule,
            "remasking_strategy_schedule": policy.remasking_strategy_schedule,
            "block_schedule": policy.block_schedule,
            "extra_step_proportions": policy.extra_step_proportions
        }

        # Evaluate the policy on the test set
        for text_prompt, tokenized_prompt, label in zip(test_formatted_questions, test_input_ids_list, test_labels):
            decoded_output, reward = evaluate_policy(model, tokenizer, tokenized_prompt, label, policy, steps, **sampling_kwargs)
            metadata[f'test_time_policy_{i}']['evals'] = {
                "prompt": text_prompt, 
                "label": label,
                "decoded_output": decoded_output, 
                "reward": reward
            }

    # Save the evaluated metadata
    with open(test_time_metadata_filename, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {test_time_metadata_filename}")
    
    # Save the tree
    with open(test_time_tree_filename, "w") as f:
        json.dump(test_time_root.to_dict(), f, indent=2)
    print(f"Saved tree to {test_time_tree_filename}")

def main():
    parser = argparse.ArgumentParser(description="GRPO-embedded MCTS over Decoding Policies")
    parser.add_argument('--mode', choices=['pretrain_from_scratch', 'pretrain_from_snapshot', 'test_time'], default='pretrain_from_scratch')
    args = parser.parse_args()

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

    # Filenames
    pre_training_metadata_filename = "pre_training_mcts_metadata.json"
    pre_training_tree_filename = "pre_training_mcts_tree_snapshot.json"

    test_time_metadata_filename = "test_time_mcts_metadata.json"
    test_time_tree_filename = "test_time_mcts_tree_snapshot.json"

    if args.mode == 'pretrain_from_scratch': # Train the tree from scratch
        print("Training tree from scratch...")
        train_tree_from_scratch(device, tokenizer, model, steps, iters, branching_factor, top_k, model_name, pre_training_metadata_filename, pre_training_tree_filename, **sampling_kwargs)
    elif args.mode == 'pretrain_from_snapshot': # Train the tree from a snapshot
        print("Training additional iterations on the tree from a snapshot...")
        num_additional_iters = 1
        train_additional_iters(num_additional_iters, device, tokenizer, model, pre_training_metadata_filename, pre_training_tree_filename)
    elif args.mode == 'test_time': # Test time GRPO-embedded MCTS
        print("Performing test time GRPO-embedded MCTS...")
        test_time_iters = 1
        test_time_grpo_embedded_mcts(test_time_iters, device, tokenizer, model, pre_training_metadata_filename, pre_training_tree_filename, test_time_metadata_filename, test_time_tree_filename)

if __name__ == '__main__':
    main()