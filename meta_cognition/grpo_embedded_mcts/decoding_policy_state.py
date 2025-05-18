import random
import torch
import torch.nn.functional as F

class DecodingPolicyState():

    def __init__(self, possible_temperatures=[0.7, 1.0], possible_remasking_strategies=["low_confidence", "random"]):
        # Options
        self.possible_temperatures = possible_temperatures
        self.possible_remasking_strategies = possible_remasking_strategies

        # Decoding policy state
        self.temperature_schedule = []
        self.remasking_strategy_schedule = []
        self.block_schedule = []
        self.extra_step_proportions = []

        self.step_id = 0
        self.block_id = 0
        self.block_end_step_id = 0

        # === Learnable logits (per decision type) ===
        self.temperature_logits = torch.nn.Parameter(torch.zeros(len(possible_temperatures)), requires_grad=True)
        self.remasking_logits = torch.nn.Parameter(torch.zeros(len(possible_remasking_strategies)), requires_grad=True)

        # Used for optimizer collection in MCTS
        self.child_logits = [self.temperature_logits, self.remasking_logits]

    def sample_partial_decoding_policy(self, steps=128, gen_length=128, max_num_blocks=4):
        """
        Sample a partial decoding policy from learnable logits.
        Updates self.*_schedule and tracks block/step information.
        """

        if self.step_id > steps:
            return

        if self.step_id == self.block_end_step_id and self.block_id < max_num_blocks:
            blocks_remaining = max_num_blocks - self.block_id

            # === Sample block length and extra step proportion ===
            if blocks_remaining == 1:
                block_length = gen_length - self.step_id
                extra_step_proportion = round(1 - sum(self.extra_step_proportions), 2)
            else:
                block_length = random.randint(1, gen_length - self.step_id - (blocks_remaining - 1))
                extra_step_proportion = round(random.uniform(0, 1 - sum(self.extra_step_proportions)), 2)

            # === Sample from learnable logits using softmax ===
            temp_probs = F.softmax(self.temperature_logits, dim=0)
            temperature_idx = torch.multinomial(temp_probs, 1).item()
            temperature = self.possible_temperatures[temperature_idx]
            self.temperature_logprob = torch.log(temp_probs[temperature_idx] + 1e-8)  # store logprob

            remask_probs = F.softmax(self.remasking_logits, dim=0)
            remasking_idx = torch.multinomial(remask_probs, 1).item()
            remasking_strategy = self.possible_remasking_strategies[remasking_idx]
            self.remasking_logprob = torch.log(remask_probs[remasking_idx] + 1e-8)  # store logprob

            # === Update state ===
            self.temperature_schedule.append(temperature)
            self.remasking_strategy_schedule.append(remasking_strategy)
            self.block_schedule.append(block_length)
            self.extra_step_proportions.append(extra_step_proportion)
            self.block_end_step_id = self.step_id + block_length
            self.block_id += 1

        self.step_id += 1