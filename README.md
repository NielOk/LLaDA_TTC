# LLaDA_TTC

This repository explores test-time compute (TTC) and denoising trajectory search (DTS) as a framework of adaptive "thinking" for LLaDA, a masked denoising language model. The goal is to design inference-time mechanisms that go beyond the original greedy remasking strategy proposed in the LLaDA paper by allowing more flexible, multi-path generation, ultimately enhancing and extending latent coherence at inference.

Original LLaDA site: https://ml-gsai.github.io/LLaDA-demo  

## Overview

LLaDA’s standard inference method remasks low-confidence tokens and refines predictions in a single trajectory. However, this can lead to irreversible early mistakes and overconfidence. This project is focused on developing a multi-trajectory inference framework that stochastically explores and refines multiple candidates across multiple remask-denoise steps. Check the scaled_inference directory for various implementations of this. The most updated, refined scaled inference scripts that implement the framework of "thinking" I am studying are centralized in scaled_inference/scaled_inference_utils.py

This repository includes:
- Denoising trajectory search (DTS) with entropy, token dependency, and noise-weighted remask scoring
- Entropy-based analysis of LLaDA on reasoning benchmarks (e.g., FOLIO)
- Pre-generation and post-generation entropy comparison scripts
- Infrastructure to run instruct/base models
- Initial experiments toward tracking resamples, entropy decay, and candidate diversity (in progress)

## Directory structure

- scaled_inference/ – core inference and scoring functions for the approaches described by this work
- entropy_experiments/ – entropy maps and analysis on FOLIO
- baseline_inference/ – original LLaDA generate.py runs (instruct/base)
- benchmark_pipeline_tests - test scripts for working with benchmarks

## Current Experiments

- Denoising Trajectory Search (DTS): maintains a beam of generation candidates, each refined over multiple remask-denoise steps. Token remasking is guided by a combination of entropy, KL/entropy change-based influence, and stochastic noise. The number of candidates retained per block is controlled by 'search_width'.

- Entropy-based analysis: includes pre- and post-generation entropy calculations, with the aim of measuring how uncertainty decays during generation and whether DTS maintains higher entropy (longer "thinking").

- Prompt-level entropy maps: being generated for the FOLIO benchmark, and future work will explore using these to dynamically allocate inference compute.

- Token dependency: estimated via KL divergence between baseline logits and logits after masking individual tokens. This score is used to bias remasking toward high-influence positions.

## Metrics (in progress)

- Number of tokens remasked that wouldn’t have been under greedy inference
- Token resample counts per position
- Unique token values sampled across candidates
- Entropy persistence during generation
- Candidate diversity and revision depth