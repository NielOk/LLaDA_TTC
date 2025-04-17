"""
Basic MCTS over a fixed-depth binary tree.
Each node in the tree can be expanded into two children (left and right),
and leaf nodes contain randomly sampled true rewards from a normal distribution.

The MCTS procedure involves:
  - Selection: traversing down the tree using UCB to balance exploration and exploitation
  - Expansion: adding children to a leaf if not yet expanded
  - Rollout: simulating a random path to a terminal leaf and sampling its reward
  - Backpropagation: updating value and visit count statistics along the visited path

After a number of simulations, the script greedily extracts the best path from root to leaf
based on the highest average value at each node. It also collects and prints statistics
for every node in the tree, including depth, number of visits, estimated value, and whether it is a leaf.
"""

import numpy as np
import random

class Node:
    def __init__(self, depth, max_depth, parent=None):
        self.parent = parent
        self.depth = depth
        self.max_depth = max_depth
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.is_leaf = depth == max_depth
        self.true_value = np.random.normal() if self.is_leaf else None

    def expand(self):
        if not self.children and not self.is_leaf:
            self.children = [
                Node(self.depth + 1, self.max_depth, parent=self),
                Node(self.depth + 1, self.max_depth, parent=self)
            ]

    def ucb_score(self, c=1.0):
        if self.visits == 0:
            return float('inf')
        return self.value + c * np.sqrt(np.log(self.parent.visits + 1) / self.visits)

def select(node):
    path = [node]
    while node.children:
        node = max(node.children, key=lambda child: child.ucb_score())
        path.append(node)
    return node, path

def rollout(node):
    # Simulate random play until reaching a leaf
    while not node.is_leaf:
        node.expand()
        node = random.choice(node.children)
    return node.true_value

def backprop(path, reward):
    for node in reversed(path):
        node.visits += 1
        node.value += (reward - node.value) / node.visits

def mcts(root, num_simulations=1000):
    for _ in range(num_simulations):
        leaf, path = select(root)
        leaf.expand()
        if leaf.children:
            leaf = random.choice(leaf.children)
            path.append(leaf)
        reward = rollout(leaf)
        backprop(path, reward)

def best_path(root):
    path = [root]
    while root.children:
        root = max(root.children, key=lambda c: c.value)
        path.append(root)
    return path

def collect_tree(node):
    all_nodes = [node]
    for child in node.children:
        all_nodes.extend(collect_tree(child))
    return all_nodes

# === Run Test ===
if __name__ == "__main__":
    np.random.seed(42)
    root = Node(depth=0, max_depth=6)

    print("Running MCTS...")
    mcts(root, num_simulations=100)

    print("\nBest Path from Root to Leaf:")
    path = best_path(root)
    for i, node in enumerate(path):
        print(f"  Depth {node.depth}: value = {round(node.value, 2)}, visits = {node.visits}")

    print(f"\nLeaf true value: {round(path[-1].true_value, 2)}")

    print("\nCollected Tree Stats:")
    all_nodes = collect_tree(root)
    for i, node in enumerate(all_nodes):
        print(f"Node {i}: depth={node.depth}, visits={node.visits}, value={round(node.value, 2)}, is_leaf={node.is_leaf}")