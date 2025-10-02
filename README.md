# Ad Rewriting with PPO LoRA and Custom Similarity Loss

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)

## Overview

This project implements PPO LoRA training with a custom similarity-based reward function for ad rewriting. The goal is to improve ad relevance for search queries while preserving important ad characteristics.

## Installation

```bash
git clone <your-repo-url>
cd ad_ppo_lora
pip install -r requirements.txt
pip install -e ".[torch,metrics]"
pip install sentence_transformers
```

## Data Requirements

### 1. RM Folder Data
Place these files in `src/llamafactory/train/rm/`:

```
src/llamafactory/train/rm/
├── rankings_original.json              # Query-ad rankings
├── sampled_ads.json               # Original ad content  
├── classified_ads.json            # Ad domain classifications
└── query_responses_original.json  # Query-response mappings
```

### 2. Data Folder
Place your training datasets in `data/` folder:
- `train_ppo.json` - PPO training data (generated using create functions from [ad-doc-reranker](https://anonymous.4open.science/r/ad-doc-reranker-57C6))
- `train_reward.json` - Reward model training data (generated using create functions from [ad-doc-reranker](https://anonymous.4open.science/r/ad-doc-reranker-57C6))
- Any other custom datasets referenced in your YAML configs

**Note**: The `train_reward` and `train_ppo` datasets are generated using the create functions from the ad-doc-reranker repository. Refer to that repository's documentation for dataset generation instructions.

## Changes Made to Repository

### Custom Similarity Loss Implementation
- **File**: `src/llamafactory/train/rm/loss.py` - Multi-component loss function (L1+L2+L3)
- **File**: `src/llamafactory/train/rm/trainer.py` - Custom reward model and trainer modifications

### Key Modifications:
1. **Custom Loss Function**: Combines query-document similarity, ranking preservation, and content preservation
2. **Data Loading**: Automatic loading of rankings, ads, and classification data
3. **Reward Calculation**: Domain-aware ad matching with custom similarity scoring
4. **Memory Optimization**: Gradient checkpointing and distributed training support

## Training Configuration

### Reward Model Settings (`llama3_lora_reward.yaml`)
Change adapter_name_or_path to sft_output folder from ad-doc-reranker

### PPO Settings (`llama3_lora_ppo.yaml`)

Change adapter_name_or_path to sft_output folder from ad-doc-reranker

## Training Pipeline

```bash
# 1. Train reward model with custom loss
llamafactory-cli train examples/train_lora/llama3_lora_reward.yaml

# 2. Train PPO with custom reward model
pip install deepspeed
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/llama3_lora_ppo.yaml
```

## Custom Loss Parameters

Located in `src/llamafactory/train/rm/trainer.py`:

```python
# Similarity Loss Configuration
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
alpha = 1.0  # Query-document similarity weight (L1)
beta = 1.0   # Triplet sampling weight (L2) 
gamma = 1.0  # Content preservation weight (L3)
use_custom_loss = True  # Enable custom loss
```

## License

This project is licensed under the Apache-2.0 License.

## Acknowledgments

- Built on [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) framework
- Uses [sentence-transformers](https://github.com/UKPLab/sentence-transformers) for text embeddings
- PPO implementation based on [TRL](https://github.com/huggingface/trl) library

---

