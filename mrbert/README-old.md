# MrBERT: BERT with MrT5-Style Delete Gates

**Author:** Hiva Mohammadzadeh  
**Course:** CS224N - Natural Language Processing with Deep Learning

This repository contains an implementation of **MrBERT**, which adapts the delete gate mechanism from the [MrT5 paper](https://arxiv.org/pdf/2410.20771) to the BERT architecture. The delete gate learns to selectively remove uninformative tokens during encoding, potentially improving computational efficiency while maintaining (or improving) task performance.

---

## Overview

The MrT5 paper introduced a **delete gate** mechanism that learns to remove redundant tokens during encoding. This can:
- **Reduce computational cost** by processing fewer tokens in later layers
- **Improve model focus** by removing uninformative tokens like punctuation or filler words
- **Potentially improve performance** by reducing noise in the representation

MrBERT adapts this idea to the encoder-only BERT architecture, enabling its use for tasks like:
- Masked Language Modeling (MLM)
- Text Classification
- Named Entity Recognition (NER)
- Question Answering
- And more...

### Key Features

- ✅ **Soft Deletion**: Masks attention scores (tokens still present, but ignored)
- ✅ **Hard Deletion**: Physically removes tokens from the sequence
- ✅ **Multiple Delete Gate Types**: Scaled sigmoid, log sigmoid, random, and fixed
- ✅ **Configurable Gate Placement**: Place the delete gate at any encoder layer
- ✅ **Gumbel Noise**: Optional exploration during training
- ✅ **Deletion Loss**: Auxiliary loss to encourage a target deletion rate
- ✅ **Full Task Support**: MLM, classification, token classification, QA, etc.

---

## Architecture

```
Input: [CLS] The quick brown fox [SEP]
         ↓
┌──────────────────────────────────────┐
│         BERT Embedding Layer          │
└──────────────────────────────────────┘
         ↓
┌──────────────────────────────────────┐
│         Encoder Layer 0               │
└──────────────────────────────────────┘
         ↓
┌──────────────────────────────────────┐
│         Encoder Layer 1               │
└──────────────────────────────────────┘
         ↓
┌──────────────────────────────────────┐
│   ★ DELETE GATE (default: Layer 2)   │  ← Learns which tokens to delete
│   ┌────────────────────────────────┐ │
│   │ LayerNorm → Linear → Sigmoid   │ │
│   └────────────────────────────────┘ │
│   Output: delete_mask per token      │
└──────────────────────────────────────┘
         ↓
   (Soft: mask attention | Hard: remove tokens)
         ↓
┌──────────────────────────────────────┐
│      Encoder Layers 2-11              │
│      (with delete mask applied)       │
└──────────────────────────────────────┘
         ↓
       Output
```
```

- **Soft Deletion**: `delete_gate_value` is added to attention scores (masking effect)
- **Hard Deletion**: Tokens with `delete_gate_value > threshold` are physically removed

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.39+

### Setup

```bash
# Clone the repository
git clone https://github.com/HivaMohammadzadeh1/CS224N-project.git
cd CS224N-project

# Create conda environment
conda create -n mrbert python=3.11 -y
conda activate mrbert

# Install dependencies
pip install torch transformers datasets accelerate tqdm matplotlib "numpy<2"

# For the original mrt5 code
cd mrt5
pip install -r requirements.txt
pip install transformers==4.39.1
```

---

## Configuration

### MrBertConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `delete_gate_layer` | int | 2 | Which encoder layer to place the delete gate (0-indexed) |
| `deletion_type` | str | "scaled_sigmoid" | Type of delete gate: `"scaled_sigmoid"`, `"log_sigmoid"`, `"random"`, `"fixed"` |
| `sigmoid_mask_scale` | float | -10.0 | Scale for sigmoid activation (more negative = stronger deletion) |
| `deletion_threshold` | float | None | Threshold for hard deletion. If None, uses soft deletion |
| `gate_layer_norm` | bool | True | Apply LayerNorm before delete gate |
| `use_gumbel_noise` | bool | False | Add Gumbel noise during training for exploration |
| `random_deletion_probability` | float | 0.5 | Deletion probability for random gate type |
| `fixed_deletion_amount` | float | 0.5 | Deletion fraction for fixed gate type |

---

## Training

```bash
# Masked Language Modeling (default)
python train_mrbert.py \
    --task mlm \
    --dataset_name wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --output_dir ./mrbert_checkpoints \
    --num_epochs 3 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --target_deletion_rate 0.3 \
    --deletion_loss_weight 0.1

# Sequence Classification (e.g., SST-2)
python train_mrbert.py \
    --task sequence_classification \
    --dataset_name glue \
    --dataset_config sst2 \
    --output_dir ./mrbert_sst2 \
    --num_epochs 3

# Token Classification (e.g., NER)
python train_mrbert.py \
    --task token_classification \
    --dataset_name conll2003 \
    --output_dir ./mrbert_ner

# Question Answering (e.g., SQuAD)
python train_mrbert.py \
    --task question_answering \
    --dataset_name squad \
    --output_dir ./mrbert_squad
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | mlm | Task: `mlm`, `sequence_classification`, `token_classification`, `question_answering` |
| `--model_name` | bert-base-uncased | Base BERT model |
| `--delete_gate_layer` | 2 | Delete gate placement |
| `--deletion_type` | scaled_sigmoid | Delete gate type |
| `--target_deletion_rate` | 0.0 | Target deletion rate (0 = no deletion loss) |
| `--deletion_loss_weight` | 0.1 | Weight for auxiliary deletion loss |
| `--delete_gate_lr` | 1e-4 | Learning rate for delete gate (can be higher than base model) |
| `--num_epochs` | 3 | Number of training epochs |
| `--batch_size` | 16 | Batch size |
| `--learning_rate` | 5e-5 | Learning rate |
| `--max_seq_length` | 128 | Maximum sequence length |
---

## Evaluation

```bash
# Evaluate a trained model
python eval_mrbert.py --model_path ./mrbert_checkpoints/final

# Evaluate on fresh MrBERT (no training)
python eval_mrbert.py --from_pretrained bert-base-uncased

# Specify evaluation dataset
python eval_mrbert.py \
    --model_path ./mrbert_checkpoints/final \
    --dataset_name wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --num_samples 1000
```

### Evaluation Metrics

The evaluation script reports:

1. **MLM Perplexity**: How well the model predicts masked tokens
2. **Deletion Rate**: Average fraction of tokens deleted
3. **Token Analysis**: Which token types are most frequently deleted
4. **Deletion by Position**: Are early/late tokens deleted more often?
5. **Comparison with BERT**: MrBERT vs baseline BERT perplexity

---

## Supported Tasks

| Model Class | Task | Output |
|-------------|------|--------|
| `MrBertModel` | Base model | Hidden states + delete gate outputs |
| `MrBertForMaskedLM` | Masked Language Modeling | Token predictions |
| `MrBertForSequenceClassification` | Text Classification | Class logits |
| `MrBertForTokenClassification` | NER, POS Tagging | Per-token labels |
| `MrBertForQuestionAnswering` | Extractive QA | Start/end positions |
| `MrBertForMultipleChoice` | Multiple Choice | Choice logits |
| `MrBertForNextSentencePrediction` | NSP | Binary classification |
---

## File Structure

```
CS224N-project/
├── modeling_mrbert.py        # Main MrBERT model implementation
├── configuration_mrbert.py   # MrBertConfig class
├── train_mrbert.py           # Training script for multiple tasks
├── eval_mrbert.py            # Evaluation script
├── test_mrbert.py            # Test suite
├── modeling_bert.py          # Reference BERT implementation
├── README.md                 # This documentation
│
├── mrbert_checkpoints/       # Saved model checkpoints
│   └── final/
│       ├── config.json
│       ├── model.safetensors
│       └── tokenizer files...
│
└── mrt5/                     # Original MrT5 reference implementation
    ├── models/
    │   └── modeling_mrt5.py  # Original T5 with delete gates
    ├── data/
    ├── eval/
    └── training/
```

---

## Testing

Run the test suite to verify the implementation:

```bash
python test_mrbert.py
```

This runs tests for:
- ✅ Configuration creation
- ✅ Delete gate modules (all types)
- ✅ MrBertModel forward pass (soft deletion)
- ✅ MrBertModel forward pass (hard deletion)
- ✅ All task-specific model heads
- ✅ Gradient flow through delete gate
- ✅ Comparison with baseline BERT
---
