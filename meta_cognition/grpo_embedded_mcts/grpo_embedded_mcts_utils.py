import copy
import random

from decoding_policy_state import DecodingPolicyState
from policy_based_decoding_utils import *
from mcts_node import MCTSNode

def expand_node(node, branching_factor, steps, **sampling_kwargs):
    '''
    **sampling_kwargs format example: 
    sampling_kwargs = {
        "possible_temperatures": [0.7, 1.0, 1.3],
        "possible_remasking_strategies": ["low_confidence", "random"],
        "steps": 128,
        "gen_length": 128,
        "max_num_blocks": 4
    }
    '''

    new_nodes = []
    for _ in range(branching_factor):
        child_state = copy.deepcopy(node.state)
        child_state.sample_partial_decoding_policy(**sampling_kwargs, steps=steps)
        child = MCTSNode(state=child_state, parent=node)
        node.children.append(child)
        new_nodes.append(child)
    return new_nodes

def rollout_policy(policy_state, steps, **sampling_kwargs):
    """
    Complete the decoding policy from its current step_id to `steps`
    by randomly sampling within the structured per-block rules.
        
    Returns a fully-sampled decoding policy.
    """

    state = copy.deepcopy(policy_state)
    while state.step_id < steps:
        state.sample_partial_decoding_policy(**sampling_kwargs)
    return state

def extract_final_answer(output):
    for line in reversed(output.strip().splitlines()):
        line = line.strip()
        if line in {"True", "False", "Uncertain"}:
            return line
    return "None"

def compute_reward(output, reference_label):
    prediction = extract_final_answer(output)
    return 1.0 if prediction.strip().lower() == reference_label.lower() else 0.0

def grpo_update_shared(children, model, prompts, labels, **sampling_kwargs):
    """
    For each child policy, evaluate it on *every* training prompt.
    Average reward across prompts, then apply GRPO across children.
    """
    child_rewards = []

    for child in children:
        completed_state = rollout_policy(child.state, **sampling_kwargs)
        total_reward = 0.0

        for prompt, reference_label in zip(prompts, labels):
            output = generate_with_decoding_policy(
                model,
                prompt,
                completed_state,
                steps=sampling_kwargs["steps"],
                gen_length=sampling_kwargs["gen_length"]
            )
            reward = compute_reward(output, reference_label)
            total_reward += reward

        avg_reward = total_reward / len(prompts)
        child.value_sum += avg_reward
        child.visits += 1
        child_rewards.append(avg_reward)

    mean_r = sum(child_rewards) / len(child_rewards)
    for child, r in zip(children, child_rewards):
        child.value_sum += (r - mean_r)  # GRPO normalization

def search_shared(model, prompts, labels, steps=128, iters=30, branching_factor=2, top_k=3, **sampling_kwargs):
    root = MCTSNode(state=DecodingPolicyState())

    for _ in range(iters):
        node = root
        while not node.is_terminal(steps) and node.is_fully_expanded(branching_factor):
            node = node.best_child()

        if not node.is_terminal(steps):
            children = expand_node(node, branching_factor, steps, **sampling_kwargs)
            grpo_update_shared(children, model, prompts, labels, **sampling_kwargs)

    # Collect all leaf nodes under the root
    all_leaves = []

    def collect_leaves(node):
        if not node.children:
            all_leaves.append(node)
        for child in node.children:
            collect_leaves(child)

    collect_leaves(root)

    # Sort and return top-k policies by average value
    top_leaves = sorted(
        all_leaves,
        key=lambda c: c.value_sum / c.visits if c.visits > 0 else float('-inf'),
        reverse=True
    )[:top_k]

    return [leaf.state for leaf in top_leaves]