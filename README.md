# LLaDA_TTC
TL;DR: We introduce a test-time inference framework—GRPO-embedded Monte Carlo Tree Search (GRPO-MCTS)—that enables text diffusion models to optimize how they decode, not just what they decode. Rather than relying on static decoding strategies, our approach performs structured search over blockwise inference schedules customized for each input prompt. 

Each decoding policy is a structured program specifying temperature, remasking strategy, step allocation, and compute distribution. GRPO-MCTS explores this policy space using Monte Carlo Tree Search, while applying Group-Relative Policy Optimization (GRPO) at each node to learn reward-aligned sampling behavior. This two-level structure, search over decoding blocks and local optimization over rollouts, forms a dynamic system of structured metacognition. Scaling is controlled by the breadth of the tree search and the precision of GRPO updates.

This work is the first to frame inference-time decoding as a real-time optimization problem over compositional strategies, demonstrating that language models can adapt how they think on a per-prompt basis. 

We gratefully acknowledge compute support provided by researchers at DeepMind in preparation for DeepMind's Gemini Diffusion release.

-----------------------------------------------------------------------------------------------------------------------------------------------
Directory Configuration:
* The paper is located at: "grpo_embedded_mcts_decoding_policies.pdf"
* The research prototype is located at: meta_cognition/grpo_embedded_mcts
* Results from the research prototype are located at: meta_cognition/grpo_embedded_mcts/trained_trees