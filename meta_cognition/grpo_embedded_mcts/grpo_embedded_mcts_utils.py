import copy

from decoding_policy_state import DecodingPolicyState
from policy_based_decoding_utils import *
from mcts_node import MCTSNode

def evaluate_policy(model, prompt, policy_state):
    output = generate_with_decoding_policy(model, prompt, policy_state)
    reward = compute_reward(output, prompt)  # You define this
    return reward

def expand_node(node, branching_factor, steps, **sampling_kwargs):
    new_nodes = []
    for _ in range(branching_factor):
        child_state = copy.deepcopy(node.state)
        child_state.sample_partial_decoding_policy(**sampling_kwargs, steps=steps)
        child = MCTSNode(state=child_state, parent=node)
        node.children.append(child)
        new_nodes.append(child)
    return new_nodes

def grpo_update(children, model, prompt):
    rewards = []
    for child in children:
        reward = evaluate_policy(model, prompt, child.state)
        child.value_sum += reward
        child.visits += 1
        rewards.append(reward)
    
    # Normalize within the group (GRPO)
    mean_r = sum(rewards) / len(rewards)
    for child, r in zip(children, rewards):
        child.value_sum += (r - mean_r)

def search(model, prompt, steps=128, iters=30, branching_factor=2, **sampling_kwargs):
    root = MCTSNode(state=DecodingPolicyState())
    for _ in range(iters):
        node = root
        while not node.is_terminal(steps) and node.is_fully_expanded(branching_factor):
            node = node.best_child()

        if not node.is_terminal(steps):
            children = expand_node(node, branching_factor, steps, **sampling_kwargs)
            grpo_update(children, model, prompt)

    # Return best full policy
    best_leaf = max(root.children, key=lambda child: child.value_sum / child.visits)
    return best_leaf.state

def iterate_prompts(model, prompts, steps=128, iters=30, branching_factor=2, **kwargs):
    best_policies = []
    for prompt in prompts:
        policy = search(model, prompt, steps=steps, iters=iters, branching_factor=branching_factor, **kwargs)
        best_policies.append(policy)
    return best_policies

def extract_final_answer(output):
    for line in reversed(output.strip().splitlines()):
        line = line.strip()
        if line in {"True", "False", "Uncertain"}:
            return line
    return "None"

def compute_reward(output, reference_label):
    # Current compute reward only checks for exact match at end of reasoning

    prediction = extract_final_answer(output)
    return 1.0 if prediction.strip().lower() == reference_label.lower() else 0.0