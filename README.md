These are some of the key papers/resources I found helpful, in improving my "reasoning" on improving LLM "reasoning". Perhaps it can do the same for someone else. 
Many thoughts, conversations and insights have melded into this list. Someday I will list down everyone else involved in this as well. For now, use it as you see fit :)

# CORE REASONING

## Key Ideas

### Reasoning Paper Chain

1. Scaling Scale: https://arxiv.org/pdf/2104.03113.pdf
2. Test-Time Compute 1(Initial GSM-8K): https://arxiv.org/pdf/2110.14168.pdf
3. Test-Time Compute 2(Path Independence): https://arxiv.org/abs/2211.09961
4. Reasoning with Reasoning STaR: https://arxiv.org/abs/2203.14465
5. Verify Step by Step: https://arxiv.org/abs/2305.20050
6. WizardLM: https://arxiv.org/abs/2304.12244
7. WizardMath: https://arxiv.org/abs/2308.09583
8. Cicero: https://noambrown.github.io/papers/22-Science-Diplomacy-TR.pdf

## RLHF Methods

## RL Resources:
1. RL Basics: https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf

### *PO & Beyond
1. PPO: https://arxiv.org/abs/1707.06347
2. DPO: https://arxiv.org/abs/2305.18290
3. IPO: https://arxiv.org/abs/2310.12036
4. C-RLFT:  https://arxiv.org/pdf/2309.11235.pdf
5. APA: https://arxiv.org/abs/2306.02231
6. https://www.interconnects.ai/p/rlhf-progress-scaling-dpo-to-70b
7. DeepMind Nash Equilibrium Method: https://misovalko.github.io/publications/munos2024nash.pdf
8. Alignment Ceiling: https://arxiv.org/abs/2311.00168

### X Threads
1. *PO Intuition: https://twitter.com/rm_rafailov/status/1729208972476059785?t=up2nqnIAx22sEmI7PZ3NvA&s=19

## Continual Pretraining
1. Base Paper: https://arxiv.org/abs/2302.03241

## Tree Searching
2. Thinking Fast and Slow: https://arxiv.org/abs/1705.08439

## Metamorphosis

### Few Shot Polymorphic Adapters
1. Base Paper: https://arxiv.org/abs/2107.04805

### LLM Pruning
1. Base Paper: https://arxiv.org/abs/2305.11627

## Prompting Based

### ReAct

1.  Base Paper: https://www.promptingguide.ai/techniques/react
2.  MultiModal ReAct: https://arxiv.org/abs/2303.11381

# SYNTHETIC DATA GENERATION

## John Durbin
1. AIroboros: https://github.com/jondurbin/airoboros

## ReST( Reinforced Self Training)

1. Base Paper: https://arxiv.org/abs/2308.08998
2. Self Instruct:  https://arxiv.org/abs/2212.10560

## Articles

1. Umbrella Intro(Nathan Lambert): https://www.interconnects.ai/p/llm-synthetic-data

# MULITIMODAL

## Embeddings

1. ImageBind: https://arxiv.org/abs/2305.05665
2. Embedding Arithmetic: https://leditsplusplus-project.static.hf.space/index.html

# QLEARNING

## Foundations
1. DeepQ: https://arxiv.org/abs/2206.01078
2. Policy Gradients Intro: https://proceedings.neurips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf
3. Policy Gradient & Q Equivalence: https://arxiv.org/abs/1704.06440
4. Rainbow(combing improvements): https://arxiv.org/abs/1710.02298

## Tutorials
QLearning(Yannic Kilcher): https://youtu.be/nOBm4aYEYR4?si=AnhFUCofB0SrngoB

# UTILITIES

## LORA and Quantization
1. LoRA: https://arxiv.org/abs/2106.09685
2. QLoRA: https://arxiv.org/abs/2305.14314
3. Quantization Scaling Laws: https://arxiv.org/pdf/2212.09720.pdf
4. Quantization: https://arxiv.org/pdf/2106.08295.pdf

## Benchmark
1. Eleuther LM Harness: https://github.com/EleutherAI/lm-evaluation-harness
2. Big Bench: https://arxiv.org/abs/2206.04615

## Swarms
1. AutoGen:  https://github.com/microsoft/autogen
2. TaskWeaver: https://arxiv.org/abs/2311.17541

## Memory Related
1. MemGPT:  https://arxiv.org/abs/2310.08560

## Reward Modelling
1. Eureka(Human Level Rewards): https://arxiv.org/abs/2310.12931

# EMOTION MODELLING

## Prompt Based
1. LLMs Respond to Emotion Stimuli: https://arxiv.org/abs/2307.11760

# EMERGENCE

## Cross Modality Emergent Behaviour
1. Palm-E: https://arxiv.org/abs/2303.03378

## Articles
1. God Neuron https://transformer-circuits.pub/2023/monosemantic-features/vis/a1.html#feature-2663
2. Emergent Memory
https://artificialintuition.substack.com/p/making-sense-of-attention

# TO PONDER

## Prompting vs Fine-Tuning
1. Big with Prompt === Small With Fine Tuning:
https://arxiv.org/pdf/2311.16452.pdf

## Is emotion modelling necessary for alignment ?

# ORIGIN STORY
1. Word2Vec: https://en.wikipedia.org/wiki/Word2vec
2. Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
3. Attention Is All You Need: https://arxiv.org/abs/1706.03762
4. nanoGPT: https://github.com/karpathy/nanoGPT
5. Transformers Catalog: https://amatriain.net/blog/transformer-models-an-introduction-and-catalog-2d1e9039f376/
6. Transformer Turing Complete: https://arxiv.org/abs/2303.14310
DeepQ: https://arxiv.org/abs/2206.01078
7. Bellman Optimality: https://www.analyticsvidhya.com/blog/2021/02/understanding-the-bellman-optimality-equation-in-reinforcement-learning/

# KEY MODELS
1. Evolved Seeker(TokenBender): https://huggingface.co/TokenBender/evolvedSeeker_1_3
2. Mistral 7B (Original): https://mistral.ai/news/announcing-mistral-7b/
3. Mistral 7b 8X: https://x.com/MistralAI/status/1733150512395038967?s=20 
4. Zephyr: https://arxiv.org/abs/2310.16944
5. FuYu: https://www.adept.ai/blog/fuyu-8b
6. Mamba: https://arxiv.org/abs/2312.00752
7. OpenHermes 2.5: https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B
8. LLaMa 2: https://ai.meta.com/llama/
9. Qwen: https://huggingface.co/Qwen
10. DeepSeek: https://huggingface.co/deepseek-ai
11. Intel Neural Chat: https://huggingface.co/Intel/neural-chat-7b-v1-1
12. Starling LM: https://starling.cs.berkeley.edu
13. OpenChat 3.5: https://huggingface.co/openchat/openchat_3.5 




# MOH MAYA (GPUs)
1. vast.ai
2. runpod.io
3. lambdalabs





















