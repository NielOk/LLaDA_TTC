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


# === Helper: Recursively collect logits from tree ===
def collect_all_child_logits(node):
    """Recursively collects all child_logits from the MCTS tree."""
    logits = []
    if hasattr(node, 'child_logits') and node.child_logits is not None:
        logits.append(node.child_logits)
    for child in node.children:
        logits.extend(collect_all_child_logits(child))
    return logits


# === Expansion ===
def expand_node(node, branching_factor, steps, **sampling_kwargs):
    '''
    **sampling_kwargs looks like this (as an example):
    {
        "possible_temperatures": [0.7, 1.0],
        "possible_remasking_strategies": ["low_confidence", "random"],
        "gen_length": 128,
        "max_num_blocks": 4
    }
    '''
    print("=== EXPANDING NODE ===")

    new_nodes = []
    for _ in range(branching_factor):
        child_state = copy.deepcopy(node.state)
        child_state.sample_partial_decoding_policy(**sampling_kwargs, steps=steps)
        child = MCTSNode(state=child_state, parent=node, branching_factor=branching_factor)
        node.children.append(child)
        new_nodes.append(child)
    return new_nodes


# === Rollout ===
def rollout_policy(policy_state, steps, **sampling_kwargs):
    print("=== ROLLING OUT POLICY ===")
    state = copy.deepcopy(policy_state)
    while state.step_id < steps:
        state.sample_partial_decoding_policy(**sampling_kwargs, steps=steps)
    print(f"Finished rollout")
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
            reward = compute_reward(decoded_output, reference_label)
            print(f"Child {i}, Prompt {j}: Reward = {reward}")
            child_rewards[i][j] = reward

    # Step 2: Per-prompt mean reward
    prompt_means = [sum(child_rewards[i][j] for i in range(n_children)) / n_children for j in range(n_prompts)]

    # Step 3: Compute total advantage per child
    advantages = []
    for i in range(n_children):
        advantage = sum(child_rewards[i][j] - prompt_means[j] for j in range(n_prompts)) / n_prompts
        advantages.append(advantage)

    # Step 4: GRPO Loss and Logit Update
    log_probs = parent.log_softmax_probs()
    loss = -sum(advantages[i] * log_probs[i] for i in range(n_children))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Step 5: Optional: Propagate value upward
    for i, child in enumerate(children):
        node = child
        while node is not None:
            node.value_sum += advantages[i]
            node.visits += 1
            node = node.parent


# === Main Search ===
def search_shared(model, tokenizer, prompts, labels, steps=128, iters=30, branching_factor=2, top_k=3, resume_node=None, **sampling_kwargs):
    if resume_node is None:
        root = MCTSNode(state=DecodingPolicyState(), branching_factor=branching_factor)
    else:
        root = resume_node

    # === Initialize persistent optimizer once over all logits ===
    all_logits = collect_all_child_logits(root)
    optimizer = torch.optim.SGD(all_logits, lr=0.1)

    for _ in range(iters):
        node = root
        while not node.is_terminal(steps) and node.is_fully_expanded(branching_factor):
            node = node.best_child()

        if not node.is_terminal(steps):
            children = expand_node(node, branching_factor, steps, **sampling_kwargs)
            print(f"[LOG] Expanded {len(children)} children at node: {node}")

            # After expanding, collect any newly created logits
            new_logits = collect_all_child_logits(root)
            if len(new_logits) > len(all_logits):
                # If new nodes with logits were added, extend optimizer param group
                new_params = [p for p in new_logits if p not in all_logits]
                optimizer.add_param_group({'params': new_params})
                all_logits = new_logits

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