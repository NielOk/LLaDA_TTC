'''
mcts node class
'''

import math

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.value_sum = 0.0
        self.visits = 0

    def is_fully_expanded(self, branching_factor):
        return len(self.children) >= branching_factor

    def is_terminal(self, steps):
        return self.state.step_id >= steps

    def ucb_score(self, c=1.0):
        if self.visits == 0:
            return float('inf')
        return (self.value_sum / self.visits) + c * math.sqrt(math.log(self.parent.visits + 1) / self.visits)

    def best_child(self):
        return max(self.children, key=lambda child: child.ucb_score())