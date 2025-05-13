import math
from decoding_policy_state import DecodingPolicyState

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.value_sum = 0.0
        self.visits = 0
        self.completed_state = None  # Store full rollout result

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

    def __repr__(self):
        return f"MCTSNode(steps={self.state.step_id}, value={self.value_sum:.2f}, visits={self.visits})"

    def to_dict(self):
        return {
            "value_sum": self.value_sum,
            "visits": self.visits,
            "decoding_policy": {
                "temperature_schedule": self.state.temperature_schedule,
                "remasking_strategy_schedule": self.state.remasking_strategy_schedule,
                "block_schedule": self.state.block_schedule,
                "extra_step_proportions": self.state.extra_step_proportions,
                "step_id": self.state.step_id,
                "block_id": self.state.block_id,
                "block_end_step_id": self.state.block_end_step_id
            },
            "children": [child.to_dict() for child in self.children]
        }

    @staticmethod
    def from_dict(data, parent=None):
        state = DecodingPolicyState()
        dp = data["decoding_policy"]
        state.temperature_schedule = dp["temperature_schedule"]
        state.remasking_strategy_schedule = dp["remasking_strategy_schedule"]
        state.block_schedule = dp["block_schedule"]
        state.extra_step_proportions = dp["extra_step_proportions"]
        state.step_id = dp["step_id"]
        state.block_id = dp["block_id"]
        state.block_end_step_id = dp["block_end_step_id"]

        node = MCTSNode(state=state, parent=parent)
        node.value_sum = data["value_sum"]
        node.visits = data["visits"]
        node.children = [MCTSNode.from_dict(child_data, parent=node) for child_data in data["children"]]
        return node