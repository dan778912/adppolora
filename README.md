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

```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3.1-8B-Instruct
adapter_name_or_path: sft_output
trust_remote_code: true

### method
stage: rm
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: train_reward
template: llama3
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/llama3-8b/lora/reward
logging_steps: 1
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none  # choices: [none, wandb, tensorboard, swanlab, mlflow]

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null
```

### PPO Settings (`llama3_lora_ppo.yaml`)

```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3.1-8B-Instruct
adapter_name_or_path: sft_output
reward_model: saves/llama3-8b/lora/reward
trust_remote_code: true

### method
stage: ppo
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all
gradient_checkpointing: true

### dataset
dataset: train_ppo
template: llama3
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 12
dataloader_num_workers: 12

### output
output_dir: saves/llama3-8b/lora/ppo
logging_steps: 1
save_steps: 500
plot_loss: true
overwrite_output_dir: true
report_to: tensorboard  # choices: [none, wandb, tensorboard, swanlab, mlflow]

### train
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 1.0e-5
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### generate
max_new_tokens: 512
top_k: 0
top_p: 0.9

### deepspeed
deepspeed:
  train_micro_batch_size_per_gpu: 4
  gradient_accumulation_steps: 4
  zero_stage: 2
  offload_optimizer:
    device: "cpu"
    pin_memory: true
  offload_param:
    device: "none"
  zero3_init_flag: false
  zero_optimization:
    stage: 2
    allgather_partitions: true
    allgather_bucket_size: 500000000
    overlap_comm: true
    reduce_scatter: true
    reduce_bucket_size: 500000000
    contiguous_gradients: true

```

## Training Pipeline

```bash
# 1. Train reward model with custom loss
llamafactory-cli train examples/train_lora/llama3_lora_reward.yaml

# 2. Train PPO with custom reward model  
llamafactory-cli train examples/train_lora/llama3_lora_ppo.yaml
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

