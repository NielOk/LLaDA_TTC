import copy
import random
import openai
import torch
from dotenv import load_dotenv
import os

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
    cloned.block_end_step_id = source_state.block_end_step_id

    # === Share logits and logprobs ===
    cloned.temperature_logits = source_state.temperature_logits
    cloned.remasking_logits = source_state.remasking_logits
    cloned.child_logits = source_state.child_logits
    cloned.temperature_logprob = getattr(source_state, "temperature_logprob", None)
    cloned.remasking_logprob = getattr(source_state, "remasking_logprob", None)

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


# === Rollout ===
def rollout_policy(policy_state, steps, **sampling_kwargs):
    print("=== ROLLING OUT POLICY ===")
    state = clone_decoding_policy_state(policy_state)
    while state.step_id < steps:
        state.sample_partial_decoding_policy(
            steps=steps,
            gen_length=sampling_kwargs["gen_length"],
            max_num_blocks=sampling_kwargs["max_num_blocks"]
        )
    print(f"Finished rollout")
    print(f"Rolled out policy: temperature schedule ({state.temperature_schedule}), remasking strategy schedule ({state.remasking_strategy_schedule}), block schedule ({state.block_schedule}), extra step proportions ({state.extra_step_proportions})")
    return state


# === Reward computation ===
def extract_final_answer(output):
    for line in reversed(output.strip().splitlines()):
        line = line.strip()
        if line in {"True", "False", "Uncertain"}:
            print(f"Extracted Answer: {line}")
            return line
    
    print("No valid answer extracted from the output.")
    return "None"


def compute_reward(prompt, output):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are scoring the output of another language model for a first order logic task. You are given a question and the model's reasoning process and answer."},
            {"role": "user", "content": f"This is the question: {prompt}. This is the model's reasoning process and answer: {output}. How many valid reasoning steps did the model take? How many relevant premises were incorporated into the reasoning? Return a vector of 2 numbers in the following format: [number_of_valid_steps, number_of_relevant_premises]"}
        ]
    )

    prediction = extract_final_answer(output)

    reward = response.choices[0].message.content.strip()
    if not reward.startswith("[") or not reward.endswith("]"):
        print(f"Invalid reward format: {reward}")
        return 0, prediction
    
    reward_0 = int(reward.split(",")[0].split("[")[1].strip())
    reward_1 = int(reward.split(",")[1].split("]")[0].strip())
    total_reward = reward_0 + reward_1
    return total_reward, prediction


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

    # Step 4: Safe GRPO loss with detached logprobs
    loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    for i in range(n_children):
        temp_logprob = children[i].temperature_logprob
        remask_logprob = children[i].remasking_logprob

        # Defensive: skip if logprobs are missing or broken
        if temp_logprob is None or remask_logprob is None:
            print(f"Child {i} has missing logprobs. Skipping...")
            continue

        total_logprob = temp_logprob + remask_logprob
        loss = loss - (advantages[i] * total_logprob)

    # Step 5: Optional: Propagate value upward
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