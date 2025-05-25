import copy
import random
import openai
import torch
from dotenv import load_dotenv
import os
import re

from decoding_policy_state import DecodingPolicyState
from policy_based_decoding_utils import *
from mcts_node import MCTSNode

# Load openai client
load_dotenv()
openai_api_key = os.getenv("niel_openai_token")
client = openai.OpenAI(api_key=openai_api_key)


# Basi
def clone_decoding_policy_state(source_state):
    from decoding_policy_state import DecodingPolicyState

    # Create new instance with same options
    cloned = DecodingPolicyState(
        possible_temperatures=source_state.possible_temperatures,
        possible_remasking_strategies=source_state.possible_remasking_strategies
    )

    # === Copy schedules and counters ===
    cloned.temperature_schedule = source_state.temperature_schedule.copy()
    cloned.remasking_strategy_schedule = source_state.remasking_strategy_schedule.copy()
    cloned.block_schedule = source_state.block_schedule.copy()
    cloned.extra_step_proportions = source_state.extra_step_proportions.copy()
    cloned.step_id = source_state.step_id
    cloned.block_id = source_state.block_id

    # === Share logits and logprobs ===
    cloned.temperature_logits = source_state.temperature_logits
    cloned.remasking_logits = source_state.remasking_logits
    cloned.child_logits = source_state.child_logits
    cloned.temperature_logprob = getattr(source_state, "temperature_logprob", None)
    cloned.remasking_logprob = getattr(source_state, "remasking_logprob", None)
    cloned.sampled_temperature_index = getattr(source_state, "sampled_temperature_index", None)
    cloned.sampled_remasking_index = getattr(source_state, "sampled_remasking_index", None)

    return cloned


# === Helper: Recursively collect logits from tree ===
def collect_all_child_logits(node):
    logits = []
    if hasattr(node.state, 'child_logits') and node.state.child_logits is not None:
        logits.extend(node.state.child_logits)
    for child in node.children:
        logits.extend(collect_all_child_logits(child))
    return logits


# === Expansion ===
def expand_node_with_pretraining(
    node, branching_factor, steps, model, tokenizer, prompts, labels, optimizer, 
    num_phase1_groups=2, rollouts_per_group=5, **sampling_kwargs
):
    if node.is_terminal(steps):
        print(f"Skipping expansion: terminal node with step_id {node.state.step_id}")
        return []

    print("=== EXPANDING NODE WITH GRPO PRETRAINING ===")
    print(f"Parent step_id = {node.state.step_id}, block_id = {node.state.block_id}")

    # === Phase 1: Multiple GRPO groups ===
    print("=== Phase 1: Multiple GRPO groups ===")
    temp_children = []
    total_rollouts = num_phase1_groups * rollouts_per_group
    for _ in range(total_rollouts):
        child_state = clone_decoding_policy_state(node.state)
        child_state.sample_partial_decoding_policy(
            steps=steps,
            gen_length=sampling_kwargs["gen_length"],
            max_num_blocks=sampling_kwargs["max_num_blocks"]
        )
        child = MCTSNode(state=child_state, parent=node, branching_factor=branching_factor)
        temp_children.append(child)

    for g in range(num_phase1_groups):
        group = temp_children[g * rollouts_per_group:(g + 1) * rollouts_per_group]
        grpo_update_per_prompt(group, model, tokenizer, prompts, labels, steps, optimizer, **sampling_kwargs)

    # === Phase 2: Final child sampling from improved logits ===
    print("=== Phase 2: Final child sampling from improved logits ===")
    final_children = []
    for _ in range(branching_factor):
        child_state = clone_decoding_policy_state(node.state)
        child_state.sample_partial_decoding_policy(
            steps=steps,
            gen_length=sampling_kwargs["gen_length"],
            max_num_blocks=sampling_kwargs["max_num_blocks"]
        )
        child = MCTSNode(state=child_state, parent=node, branching_factor=branching_factor)
        node.children.append(child)
        final_children.append(child)

    return final_children


def rollout_policy(policy_state, steps, **sampling_kwargs):
    print("=== ROLLING OUT POLICY ===")
    state = clone_decoding_policy_state(policy_state)

    while state.step_id < steps and state.block_id < sampling_kwargs["max_num_blocks"]:
        state.sample_partial_decoding_policy(
            steps=steps,
            gen_length=sampling_kwargs["gen_length"],
            max_num_blocks=sampling_kwargs["max_num_blocks"]
        )

    print("Finished rollout")
    print(f"Rolled out policy:")
    print(f"  temperature schedule: {state.temperature_schedule}")
    print(f"  remasking strategy schedule: {state.remasking_strategy_schedule}")
    print(f"  block schedule: {state.block_schedule}")
    print(f"  extra step proportions: {state.extra_step_proportions}")

    return state


# === Reward Computation ===
def extract_final_answer(output):
    # Match 'True', 'False', or 'Uncertain' near the end, even with markdown or punctuation
    match = re.search(r'(True|False|Uncertain)\s*(\*\*)?[\.!\s]*$', output.strip(), re.IGNORECASE)
    if match:
        final = match.group(1).capitalize()
        print(f"Extracted Answer: {final}")
        return final

    print("No valid answer extracted from the output.")
    return "None"


def compute_reward(prompt, output):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are scoring the output of another language model for a first order logic task. You are given a question and the model's reasoning process and answer."},
            {"role": "user", "content": f"This is the question: {prompt}. This is the model's reasoning process and answer: {output}. How many valid reasoning steps did the model take? How many relevant premises were incorporated into the reasoning? Return a vector of 2 numbers in the following format, including the square brackets, making sure it is the absolute last piece of text you output: [number_of_valid_steps, number_of_relevant_premises]"}
        ]
    )

    prediction = extract_final_answer(output)
    content = response.choices[0].message.content.strip()

    # Use regex to extract the last bracketed pair of integers
    match = re.search(r"\[(\d+),\s*(\d+)\]\s*$", content)
    if match:
        reward_0 = int(match.group(1))
        reward_1 = int(match.group(2))
        total_reward = reward_0 + reward_1
        return total_reward, prediction
    else:
        print(f"Invalid reward format: {content}")
        return 0, prediction


# === GRPO Update with Reward Propagation ===
def grpo_update_per_prompt(children, model, tokenizer, prompts, labels, steps, optimizer, **sampling_kwargs):
    '''
    For each child decoding policy, evaluate it on each prompt.
    Then normalize the reward per prompt and compute the total advantage.
    Apply GRPO by computing gradient loss from log softmax and advantages.
    '''
    print("=== GRPO Update ===")
    n_prompts = len(prompts)
    n_children = len(children)
    parent = children[0].parent

    # Matrix: child_rewards[i][j] = reward of child i on prompt j
    child_rewards = [[0.0 for _ in range(n_prompts)] for _ in range(n_children)]

    # Step 1: Complete all decoding policies
    for i, child in enumerate(children):
        completed_state = rollout_policy(child.state, steps=steps, **sampling_kwargs)
        child.completed_state = completed_state  # store it for reuse

        for j, (prompt, reference_label) in enumerate(zip(prompts, labels)):
            print(f"Child {i}, Prompt {j}: Evaluating...")
            output = generate_with_decoding_policy(
                model,
                prompt,
                completed_state,
                steps=steps,
                gen_length=sampling_kwargs['gen_length']
            )
            decoded_output = tokenizer.batch_decode(output[:, prompt.shape[1]:], skip_special_tokens=True)[0]
            print(f"Child {i}, Prompt {j}: Decoded output: {decoded_output}")
            total_reward, prediction = compute_reward(prompt, decoded_output)
            print(f"Child {i}, Prompt {j}: Reward = {total_reward}, Prediction = {prediction}")
            child_rewards[i][j] = total_reward

    # Step 2: Per-prompt mean reward
    prompt_means = [sum(child_rewards[i][j] for i in range(n_children)) / n_children for j in range(n_prompts)]

    # Step 3: Compute total advantage per child
    advantages = []
    for i in range(n_children):
        advantage = sum(child_rewards[i][j] - prompt_means[j] for j in range(n_prompts)) / n_prompts
        advantages.append(advantage)

    # Step 4: Normalize advantages to mean 0, std 1
    mean_adv = sum(advantages) / len(advantages)
    std_adv = (sum((a - mean_adv) ** 2 for a in advantages) / len(advantages)) ** 0.5
    std_adv = max(std_adv, 1e-6)  # avoid divide-by-zero

    advantages = [(a - mean_adv) / std_adv for a in advantages]

    # Step 5: Safe GRPO loss with recomputed logprobs
    losses = []

    for i in range(n_children):
        try:
            logits_temp = children[i].state.temperature_logits
            idx_temp = int(children[i].state.sampled_temperature_index)
            if isinstance(idx_temp, torch.Tensor):
                idx_temp = idx_temp.item()  # convert to int

            temp_logprobs = torch.log_softmax(logits_temp, dim=-1)
            temp_logprob = temp_logprobs[idx_temp]  # scalar

            logits_remask = children[i].state.remasking_logits
            idx_remask = int(children[i].state.sampled_remasking_index)
            if isinstance(idx_remask, torch.Tensor):
                idx_remask = idx_remask.item()  # convert to int

            remask_logprobs = torch.log_softmax(logits_remask, dim=-1)
            remask_logprob = remask_logprobs[idx_remask]  # scalar

            total_logprob = temp_logprob + remask_logprob
            loss_term = -advantages[i] * total_logprob
            losses.append(loss_term)

        except Exception as e:
            print(f"Child {i}: skipping due to logprob error: {e}")
            continue

    # Backprop if at least one child contributed
    if losses:
        total_loss = torch.stack(losses).sum()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # Step 6: Propagate value upward
    for i, child in enumerate(children):
        node = child
        while node is not None:
            node.value_sum += advantages[i]
            node.visits += 1
            node = node.parent


# === Main Search ===
def search_shared(model, tokenizer, prompts, labels, steps=128, iters=30, branching_factor=2, top_k=3, num_phase1_groups=2, rollouts_per_group=5, resume_node=None, **sampling_kwargs):
    if resume_node is None:
        root = MCTSNode(
            state=DecodingPolicyState(
                possible_temperatures=sampling_kwargs["possible_temperatures"],
                possible_remasking_strategies=sampling_kwargs["possible_remasking_strategies"]
            ),
            branching_factor=branching_factor
        )
    else:
        root = resume_node

    # === Initialize persistent optimizer once over all logits ===
    all_logits = collect_all_child_logits(root)
    optimizer = torch.optim.SGD(all_logits, lr=0.1)
    known_param_ids = {id(p) for p in all_logits}

    for _ in range(iters):
        node = root
        while not node.is_terminal(steps) and node.is_fully_expanded(branching_factor):
            node = node.best_child()

        if not node.is_terminal(steps):
            children = expand_node_with_pretraining(
                node, branching_factor, steps,
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                labels=labels,
                optimizer=optimizer,
                num_phase1_groups=num_phase1_groups,
                rollouts_per_group=rollouts_per_group,
                **sampling_kwargs
            )
            print(f"[LOG] Expanded {len(children)} children at node: {node}")

            # After expanding, collect any newly created logits
            new_logits = collect_all_child_logits(root)
            new_params = [p for p in new_logits if id(p) not in known_param_ids]

            if new_params:
                optimizer.add_param_group({'params': new_params})
                known_param_ids.update(id(p) for p in new_params)
                all_logits.extend(new_params)

            grpo_update_per_prompt(children, model, tokenizer, prompts, labels, steps, optimizer, **sampling_kwargs)

    # === Collect and rank final leaf nodes ===
    all_leaves = []
    def collect_leaves(node):
        if not node.children:
            all_leaves.append(node)
        for child in node.children:
            collect_leaves(child)

    collect_leaves(root)

    top_leaves = sorted(
        all_leaves,
        key=lambda c: c.value_sum / c.visits if c.visits > 0 else float('-inf'),
        reverse=True
    )[:top_k]

    return root, [leaf.completed_state for leaf in top_leaves]