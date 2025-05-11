import copy
import random
import openai
from dotenv import load_dotenv
import os

from decoding_policy_state import DecodingPolicyState
from policy_based_decoding_utils import *
from mcts_node import MCTSNode

# Load openai client
load_dotenv()
openai_api_key = os.getenv("niel_openai_token")
client = openai.OpenAI(api_key=openai_api_key)

# === Expansion ===
def expand_node(node, branching_factor, steps, **sampling_kwargs):
    '''
    **sampling_kwargs looks like this (as an example):
    {
        "possible_temperatures": [0.7, 1.0],
        "possible_remasking_strategies": ["low_confidence", "random"],
        "steps": 256,
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


# === Rollout ===
def rollout_policy(policy_state, steps, **sampling_kwargs):
    state = copy.deepcopy(policy_state)
    while state.step_id < steps:
        state.sample_partial_decoding_policy(**sampling_kwargs)
    return state


# === Reward computation ===
def extract_final_answer(output):
    for line in reversed(output.strip().splitlines()):
        line = line.strip()
        if line in {"True", "False", "Uncertain"}:
            return line
    return "None"


def compute_reward(output, reference_label):
    prediction = extract_final_answer(output)
    return 1.0 if prediction.strip().lower() == reference_label.lower() else 0.0


# === GRPO Update with Reward Propagation ===
def grpo_update_per_prompt(children, model, prompts, labels, **sampling_kwargs):
    """
    For each child decoding policy, evaluate it on each prompt.
    Then normalize the reward per prompt and compute the total advantage.
    Apply GRPO by propagating that advantage up the tree.
    """
    n_prompts = len(prompts)
    n_children = len(children)

    # Matrix: child_rewards[i][j] = reward of child i on prompt j
    child_rewards = [[0.0 for _ in range(n_prompts)] for _ in range(n_children)]

    # Step 1: Complete all decoding policies
    for i, child in enumerate(children):
        completed_state = rollout_policy(child.state, **sampling_kwargs)
        child.completed_state = completed_state  # store it for reuse

        for j, (prompt, reference_label) in enumerate(zip(prompts, labels)):
            output = generate_with_decoding_policy(
                model,
                prompt,
                completed_state,
                steps=sampling_kwargs["steps"],
                gen_length=sampling_kwargs["gen_length"]
            )
            reward = compute_reward(output, reference_label)
            child_rewards[i][j] = reward

    # Step 2: Per-prompt mean reward
    prompt_means = []
    for j in range(n_prompts):
        mean_rj = sum(child_rewards[i][j] for i in range(n_children)) / n_children
        prompt_means.append(mean_rj)

    # Step 3: Compute total advantage per child
    for i, child in enumerate(children):
        # Sum of prompt-wise advantages
        total_advantage = sum(
            child_rewards[i][j] - prompt_means[j]
            for j in range(n_prompts)
        )

        # Normalize by number of prompts
        total_advantage /= n_prompts

        # Step 4: Propagate advantage up the tree
        node = child
        while node is not None:
            node.value_sum += total_advantage
            node.visits += 1
            node = node.parent


# === Main Search ===
def search_shared(model, prompts, labels, steps=128, iters=30, branching_factor=2, top_k=3, **sampling_kwargs):
    root = MCTSNode(state=DecodingPolicyState())

    for _ in range(iters):
        node = root
        while not node.is_terminal(steps) and node.is_fully_expanded(branching_factor):
            node = node.best_child()

        if not node.is_terminal(steps):
            children = expand_node(node, branching_factor, steps, **sampling_kwargs)
            grpo_update_per_prompt(children, model, prompts, labels, **sampling_kwargs)

    # Collect all leaf nodes
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

    return [leaf.completed_state for leaf in top_leaves]  # return best decoding policies