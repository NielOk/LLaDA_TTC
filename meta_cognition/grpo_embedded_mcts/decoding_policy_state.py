import random
import torch
import torch.nn.functional as F

class DecodingPolicyState():

    def __init__(self, possible_temperatures=[0.0, 0.1, 0.2], possible_remasking_strategies=["low_confidence", "random"]):
        # Options
        self.possible_temperatures = possible_temperatures
        self.possible_remasking_strategies = possible_remasking_strategies

        # Decoding policy state
        self.temperature_schedule = []
        self.remasking_strategy_schedule = []
        self.block_schedule = []
        self.extra_step_proportions = []
        self.sampled_temperature_index = None
        self.sampled_remasking_index = None

        self.step_id = 0
        self.block_id = 0

        # === Learnable logits (per decision type) ===
        self.temperature_logits = torch.nn.Parameter(torch.zeros(len(possible_temperatures)), requires_grad=True)
        self.remasking_logits = torch.nn.Parameter(torch.zeros(len(possible_remasking_strategies)), requires_grad=True)

        # Used for optimizer collection in MCTS
        self.child_logits = [self.temperature_logits, self.remasking_logits]

    def sample_partial_decoding_policy(self, steps=128, gen_length=128, max_num_blocks=4):
        """
        Sample one full decoding block: temperature, remasking strategy, block length, and extra step proportion.
        Advances step_id by the block length (not one step at a time).
        Appends sampled values to schedules and updates internal counters.
        """

        # Do nothing if generation is complete or max blocks reached
        if self.step_id >= steps or self.block_id >= max_num_blocks:
            return

        blocks_remaining = max_num_blocks - self.block_id
        remaining_steps = gen_length - self.step_id

        # === Sample block length and extra step proportion ===
        if blocks_remaining == 1:
            block_length = remaining_steps
            extra_step_proportion = round(1.0 - sum(self.extra_step_proportions), 2)
        else:
            max_length = remaining_steps - (blocks_remaining - 1)
            block_length = random.randint(1, max_length)
            max_extra = 1.0 - sum(self.extra_step_proportions)
            extra_step_proportion = round(random.uniform(0, max_extra), 2)

        # === Sample temperature ===
        temp_probs = F.softmax(self.temperature_logits, dim=0)
        temperature_idx = torch.multinomial(temp_probs, 1).item()
        temperature = self.possible_temperatures[temperature_idx]
        self.temperature_logprob = torch.log(temp_probs[temperature_idx] + 1e-8)

        # === Sample remasking strategy ===
        remask_probs = F.softmax(self.remasking_logits, dim=0)
        remasking_idx = torch.multinomial(remask_probs, 1).item()
        remasking_strategy = self.possible_remasking_strategies[remasking_idx]
        self.remasking_logprob = torch.log(remask_probs[remasking_idx] + 1e-8)

        # === Update schedules and state ===
        self.temperature_schedule.append(temperature)
        self.remasking_strategy_schedule.append(remasking_strategy)
        self.block_schedule.append(block_length)
        self.extra_step_proportions.append(extra_step_proportion)
        self.sampled_temperature_index = temperature_idx
        self.sampled_remasking_index = remasking_idx

        # === Advance step and block counters ===
        self.step_id += block_length
        self.block_id += 1