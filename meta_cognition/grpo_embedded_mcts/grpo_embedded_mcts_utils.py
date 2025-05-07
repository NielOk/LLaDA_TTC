import copy

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

def grpo_update(children, model, prompt, reference_label, **sampling_kwargs):
    rewards = []

    for child in children:
        # Rollout the child policy to full length
        completed_state = rollout_policy(child.state, **sampling_kwargs)

        # Evaluate the rollout-completed policy
        output = generate_with_decoding_policy(
            model,
            prompt,
            completed_state,
            steps=sampling_kwargs["steps"],
            gen_length=sampling_kwargs["gen_length"]
        )
        reward = compute_reward(output, reference_label)

        # Update the node stats
        child.value_sum += reward
        child.visits += 1
        rewards.append(reward)

    # GRPO: group-relative normalization
    mean_r = sum(rewards) / len(rewards)
    for child, r in zip(children, rewards):
        child.value_sum += (r - mean_r)  # relative to sibling mean

def search(model, prompt, reference_label, steps=128, iters=30, branching_factor=2, **sampling_kwargs):
    root = MCTSNode(state=DecodingPolicyState())

    for _ in range(iters):
        node = root
        while not node.is_terminal(steps) and node.is_fully_expanded(branching_factor):
            node = node.best_child()

        if not node.is_terminal(steps):
            children = expand_node(node, branching_factor, steps, **sampling_kwargs)
            grpo_update(children, model, prompt, reference_label, **sampling_kwargs)

    # Return best child of root (could generalize to full tree search)
    best_leaf = max(root.children, key=lambda child: child.value_sum / child.visits)
    return best_leaf.state

def iterate_prompts(model, prompts, labels, steps=128, iters=30, branching_factor=2, **kwargs):
    best_policies = []
    for prompt, label in zip(prompts, labels):
        policy = search(model, prompt, reference_label=label, steps=steps, iters=iters, branching_factor=branching_factor, **kwargs)
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