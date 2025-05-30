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

        formatted_question = f'The following facts are given: {premise}. Is the conclusion "{conclusion}" logically entailed by the above facts? Think out loud step by step and, on a new line, write one of the following three options by itself as your final answer: False, Uncertain, or True'
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

def test_time_grpo_embedded_mcts(num_folio_questions_to_sample_test, 
    device, 
    tokenizer, 
    model, 
    steps, 
    iters, 
    branching_factor, 
    top_k, 
    num_phase1_groups,
    rollouts_per_group,
    model_name, 
    metadata_filename, 
    tree_filename,
    **sampling_kwargs
    ):
    """
    Perform GRPO-embedded MCTS over decoding polciies from scratch"""

    # Get train and test sets
    test_formatted_questions, test_labels = load_folio_train_dataset(num_questions_to_sample=num_folio_questions_to_sample_test)
    
    # Convert training set to input ids
    test_input_ids_list = convert_prompts_to_input_ids(test_formatted_questions, tokenizer, device)

    root, top_policies = search_shared(
        model=model, 
        tokenizer=tokenizer,
        prompts=test_input_ids_list,
        labels=test_labels,
        steps=steps,
        iters=iters,
        branching_factor=branching_factor,
        top_k=top_k,
        num_phase1_groups=num_phase1_groups,
        rollouts_per_group=rollouts_per_group,
        **sampling_kwargs,
    )

    # Save the top policies and metadata
    top_policies_dict = {}
    top_policies_dict['metadata'] = {
        "description": "Top policies and metadata from GRPO-embedded MCTS over decoding policies pretraining",
        "model_name": model_name,
        "num_questions_sampled_for_testing": num_folio_questions_to_sample_test,
        "folio_test_dataset_prompts": test_formatted_questions,
        "folio_test_dataset_labels": test_labels,
        "steps": steps,
        "iters": iters,
        "branching_factor": branching_factor,
        "top_k": top_k,
        "num_phase1_groups": num_phase1_groups,
        "rollouts_per_group": rollouts_per_group,
        "possible_temperatures": sampling_kwargs["possible_temperatures"],
        "possible_remasking_strategies": sampling_kwargs["possible_remasking_strategies"],
        "gen_length": sampling_kwargs["gen_length"],
        "max_num_blocks": sampling_kwargs["max_num_blocks"]
    }
    for i, policy in enumerate(top_policies):
        top_policies_dict[f"test_time_policy_{i}"] = {
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
    with open(tree_filename, "w") as f:
        json.dump(root.to_dict(), f, indent=2)
    print(f"Saved tree to {tree_filename}")

def evaluate_policy(model, tokenizer, prompt, label, policy, steps, **sampling_kwargs):
    '''
    Evaluate a single policy on a single prompt and label and score it.
    '''
    output = generate_with_decoding_policy(model, prompt, policy, steps=steps, gen_length=sampling_kwargs['gen_length'])
    decoded_output = tokenizer.batch_decode(output[:, prompt.shape[1]:], skip_special_tokens=True)[0]

    print(f"Decoded output: {decoded_output}")
    print(f"Label: {label}")
    total_reward, prediction = compute_reward(prompt, decoded_output)
    print(f"Reward: {total_reward}")

    return decoded_output, total_reward, prediction

def main():
    device = 'cuda'

    print("started model loading...")

    model_name = 'GSAI-ML/LLaDA-8B-Instruct' # Use instruct model by default

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

    print("Model loaded successfully.")

    # Model parameters
    steps = 256
    iters = 20
    branching_factor = 2 # number of children to sample at each node
    top_k = 3
    possible_temperatures = [0.1, 0.0, 0.2]
    possible_remasking_strategies = ["low_confidence", "random"]
    gen_length = 128
    max_num_blocks = 4 # Depth of the tree
    num_phase1_groups = 2
    rollouts_per_group = 5

    # Dataset parameters
    num_folio_questions_to_sample_test = 1 # Try smaller set for testing. basically, hyper-optimizing to 1 question.

    for i in range(num_folio_questions_to_sample_test):
        print(f"Sampled question {i+1} of {num_folio_questions_to_sample_test}")

        sampling_kwargs = {
            "possible_temperatures": possible_temperatures,
            "possible_remasking_strategies": possible_remasking_strategies,
            "gen_length": gen_length,
            "max_num_blocks": max_num_blocks,
        }

        # Filenames
        test_time_metadata_filename = f"test_time_grpo_embedded_mcts_metadata_{i}.json"
        test_time_tree_filename = f"test_time_grpo_embedded_mcts_tree_snapshot_{i}.json"

        print("Training tree from scratch...")
        num_questions_run = 1
        test_time_grpo_embedded_mcts(
            num_questions_run, 
            device, 
            tokenizer, 
            model, 
            steps, 
            iters, 
            branching_factor,
            top_k, 
            num_phase1_groups,
            rollouts_per_group,
            model_name, 
            test_time_metadata_filename, 
            test_time_tree_filename, 
            **sampling_kwargs
        )

    for i in range(num_folio_questions_to_sample_test):
        test_time_metadata_filename = f"test_time_grpo_embedded_mcts_metadata_{i}.json"
        test_time_tree_filename = f"test_time_grpo_embedded_mcts_tree_snapshot_{i}.json"

        ## Evaluate the top policy ##
        with open(test_time_metadata_filename, "r") as f:
            metadata = json.load(f)
        with open(test_time_tree_filename, "r") as f:
            tree = json.load(f)
        prompt = metadata['metadata']['folio_test_dataset_prompts'][0]
        label = metadata['metadata']['folio_test_dataset_labels'][0]
        print(f"Prompt: {prompt}")
        prompt = convert_prompts_to_input_ids([prompt], tokenizer, device)[0]
        top_policy = DecodingPolicyState(
            possible_temperatures=metadata['metadata']['possible_temperatures'],
            possible_remasking_strategies=metadata['metadata']['possible_remasking_strategies']
        )
        top_policy.temperature_schedule = metadata[f"test_time_policy_0"]["temperature_schedule"]
        top_policy.remasking_strategy_schedule = metadata[f"test_time_policy_0"]["remasking_strategy_schedule"]
        top_policy.block_schedule = metadata[f"test_time_policy_0"]["block_schedule"]
        top_policy.extra_step_proportions = metadata[f"test_time_policy_0"]["extra_step_proportions"]
        sampling_kwargs = {
            "possible_temperatures": metadata['metadata']['possible_temperatures'],
            "possible_remasking_strategies": metadata['metadata']['possible_remasking_strategies'],
            "gen_length": metadata['metadata']['gen_length'],
            "max_num_blocks": metadata['metadata']['max_num_blocks'],
        }

        decoded_output, total_reward, prediction = evaluate_policy(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            label=label,
            policy=top_policy,
            steps=steps,
            **sampling_kwargs
        )

        print(f"Decoded output: {decoded_output}")
        print(f"Prediction: {prediction}")
        print(f"Label: {label}")
        print(f"Reward: {total_reward}")

if __name__ == '__main__':
    main()