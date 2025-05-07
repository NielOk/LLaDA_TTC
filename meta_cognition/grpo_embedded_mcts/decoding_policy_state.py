'''
decoding policy state class object
'''

import random

class DecodingPolicyState():

    def __init__(self):
        # Decoding policy state
        self.temperature_schedule = []
        self.remasking_strategy_schedule = []
        self.block_schedule = []
        self.extra_step_proportions = []
        self.step_id = 0
        self.block_id = 0
        self.block_end_step_id = 0


    def sample_partial_decoding_policy(self, possible_temperatures=[], possible_remasking_strategies=["low_confidence", "random"], steps=128, gen_length=128, max_num_blocks=4):
        """
        Sample a partial decoding policy from the space of possible remaining 
        decoding policies and update the decoding policy state.
        """

        # Sample a new partial policy state
        if self.step_id < steps:

            # Sample a block length and extra step proportion if eligible
            if self.step_id == self.block_end_step_id and self.block_id < max_num_blocks:
                blocks_remaining = max_num_blocks - self.block_id

                # Sample a block length from the possible block lengths
                if blocks_remaining == 1: # Scenario where we only have 1 block remaining when time to sample a new block
                    block_length = gen_length - self.step_id
                    extra_step_proportion = round(1 - sum(self.extra_step_proportions), 2)
                else: # Scenario where we have multiple blocks remaining
                    block_length = random.randint(1, gen_length - self.step_id - (blocks_remaining - 1))
                    extra_step_proportion = round(random.uniform(0, 1 - sum(self.extra_step_proportions)), 2)

                # Update decoding policy state
                temperature = random.choice(possible_temperatures)
                self.temperature_schedule.append(temperature)
                remasking_strategy = random.choice(possible_remasking_strategies)
                self.remasking_strategy_schedule.append(remasking_strategy)


                self.block_end_step_id = self.step_id + block_length
                self.block_schedule.append(block_length)
                self.extra_step_proportions.append(extra_step_proportion)
                self.block_id += 1
        else:
            print("All steps have been sampled. No more partial decoding policies can be sampled.")
            return
        
        self.step_id += 1
        return