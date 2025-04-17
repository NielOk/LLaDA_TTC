import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

class Node:
    def __init__(self, depth, max_depth, parent=None):
        self.parent = parent
        self.depth = depth
        self.max_depth = max_depth
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.id = None

        if depth < max_depth:
            self.theta = torch.nn.Parameter(torch.zeros(2))  # Learnable logits for GRPO
        else:
            self.true_value = np.random.normal()
            self.is_leaf = True
            self.theta = None
        self.baseline = 0.0
        self.is_leaf = depth == max_depth

    def expand(self):
        if not self.children and not self.is_leaf:
            self.children = [
                Node(self.depth + 1, self.max_depth, parent=self),
                Node(self.depth + 1, self.max_depth, parent=self)
            ]

    def get_policy(self):
        if self.theta is None:
            return None
        return torch.distributions.Categorical(logits=self.theta)

    def choose_action(self):
        dist = self.get_policy()
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
    
# To get full tree structure
def expand_full_tree(node):
    if not node.is_leaf:
        node.expand()
        expand_full_tree(node.children[0])
        expand_full_tree(node.children[1])

def rollout_and_grpo(root):
    node = root
    path = []
    log_probs = []

    while not node.is_leaf:
        node.expand()
        action, log_prob = node.choose_action()
        log_probs.append(log_prob)
        path.append(node)
        node = node.children[action]
    path.append(node)
    return path, log_probs, node.true_value

def backprop_grpo(path, log_probs, reward):
    for node, log_prob in zip(path, log_probs):
        advantage = reward - node.baseline
        loss = -log_prob * advantage
        loss.backward()
        node.baseline = 0.9 * node.baseline + 0.1 * reward

    # Update values and visits
    for node in path:
        node.visits += 1
        node.value += (reward - node.value) / node.visits

def collect_params(node):
    if node.is_leaf:
        return []
    return [node.theta] + collect_params(node.children[0]) + collect_params(node.children[1])

def train(root, episodes=1000, lr=0.05):
    all_params = [p for p in collect_params(root)]
    optimizer = optim.SGD(all_params, lr=lr)

    for _ in range(episodes):
        optimizer.zero_grad()
        path, log_probs, reward = rollout_and_grpo(root)
        backprop_grpo(path, log_probs, reward)
        optimizer.step()

def best_path(root):
    path = [root]
    while not root.is_leaf:
        root = max(root.children, key=lambda c: c.value)
        path.append(root)
    return path

def assign_node_ids(node, counter=[0]):
    node.id = counter[0]
    counter[0] += 1
    for child in node.children:
        assign_node_ids(child, counter)

def collect_tree(node):
    all_nodes = [node]
    for child in node.children:
        all_nodes.extend(collect_tree(child))
    return all_nodes

# === Run ===
if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    max_depth = 6
    root = Node(depth=0, max_depth=max_depth)
    assign_node_ids(root)
    expand_full_tree(root)

    print("Running GRPO-based MCTS...\n")
    train(root, episodes=1000)

    print("Best Path from Root to Leaf:")
    path = best_path(root)
    for node in path:
        print(f"  Depth {node.depth}: value = {node.value:.2f}, visits = {node.visits}")

    print(f"\nLeaf true value: {path[-1].true_value:.2f}")

    print("\nCollected Tree Stats:")
    all_nodes = collect_tree(root)
    for node in all_nodes:
        val = node.true_value if node.is_leaf else node.value
        print(f"Node {node.id}: depth={node.depth}, visits={node.visits}, "
              f"value={val:.2f}, is_leaf={node.is_leaf}")