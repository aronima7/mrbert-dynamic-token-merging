# Per-Example Deletion Rate vs Loss Analysis

This analysis helps answer: **Does MrBERT make smart decisions about which tokens to delete?**

## Key Question

If a model deletes 50% of tokens on average, does it have higher loss on examples where it deleted 70% of tokens? A weak or negative correlation would indicate the model is choosing to delete tokens wisely.

## Quick Start

### For Sequence Classification (e.g., SST-2):
```bash
python analyze_deletion_correlation.py \
    --model_path ./mrbert_sst2/final \
    --task sequence_classification \
    --dataset_name glue \
    --dataset_config sst2 \
    --output_dir ./analysis_sst2
```

### For Masked Language Modeling (e.g., WikiText):
```bash
python analyze_deletion_correlation.py \
    --model_path ./mrbert_wikitext/final \
    --task mlm \
    --dataset_name wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --output_dir ./analysis_wikitext
```

### For Question Answering (e.g., SQuAD):
```bash
python analyze_deletion_correlation.py \
    --model_path ./mrbert_squad/final \
    --task question_answering \
    --dataset_name squad \
    --output_dir ./analysis_squad
```

### For TyDi QA (multilingual):
```bash
python analyze_deletion_correlation.py \
    --model_path ./mrbert_tydiqa/final \
    --task question_answering \
    --dataset_name tydiqa \
    --dataset_config primary_task \
    --output_dir ./analysis_tydiqa \
    --max_samples 1000
```

## Options

- `--model_path`: Path to trained MrBERT checkpoint (required)
- `--task`: Task type - `mlm`, `sequence_classification`, `token_classification`, or `question_answering`
- `--dataset_name`: HuggingFace dataset name (e.g., `glue`, `wikitext`, `squad`)
- `--dataset_config`: Dataset subset (e.g., `sst2`, `wikitext-2-raw-v1`)
- `--max_samples`: Limit analysis to N examples (useful for quick tests)
- `--batch_size`: Batch size for evaluation (default: 8)
- `--split`: Dataset split to analyze - `validation` or `test` (default: `validation`)
- `--output_dir`: Where to save results and plots (default: `./deletion_analysis`)

## What You Get

### Console Output
- **Deletion rate statistics**: mean, median, std, min, max (as percentages)
- **Loss statistics**: mean, median, std, min, max
- **Correlation metrics**:
  - Pearson correlation coefficient and p-value
  - Spearman rank correlation coefficient and p-value
  - Interpretation (very weak / weak / moderate / strong / very strong)
- **Binned analysis**: Mean loss for examples grouped by deletion rate ranges (0-30%, 30-50%, 50-70%, 70-100%)

### Visualizations (saved as PNG files)
1. **`deletion_vs_loss_scatter.png`**: Scatter plot showing each example's deletion rate vs loss, with a linear regression line
2. **`deletion_bins_mean_loss.png`**: Bar chart showing mean loss for different deletion rate bins
3. **`deletion_rate_histogram.png`**: Distribution of deletion rates across all examples
4. **`deletion_vs_loss_2dhist.png`**: 2D histogram showing the joint distribution

### JSON Results
**`analysis_results.json`**: Machine-readable summary with all statistics, correlation values, and binned analysis results

## Interpreting Results

### Strong Positive Correlation (r > 0.5)
⚠️ **Warning**: Examples with higher deletion rates have substantially higher losses. The model may be deleting important tokens.

### Weak Correlation (|r| < 0.3)
✓ **Good**: The model is making smart deletion decisions! Examples with higher deletion don't suffer proportionally higher losses.

### Example Output:
```
Correlation Analysis:
  Pearson  r = 0.1234  (p = 1.23e-05)
  Spearman r = 0.1456  (p = 3.45e-07)

  → WEAK positive correlation

  ✓ This suggests the model is making SMART deletion decisions!
    Examples with higher deletion rates don't have substantially higher losses.

Binned Analysis:
  Deletion 0-30%:    n= 450  mean_loss=0.3421  std=0.1234
  Deletion 30-50%:   n= 892  mean_loss=0.3508  std=0.1289
  Deletion 50-70%:   n= 345  mean_loss=0.3612  std=0.1401
  Deletion 70-100%:  n=  45  mean_loss=0.3789  std=0.1523
```

In this example:
- The correlation is weak (r ≈ 0.12), meaning deletion rate doesn't strongly predict loss
- As deletion increases, mean loss increases only slightly (0.34 → 0.38)
- This indicates the model successfully identifies less important tokens to delete

## Quick Test

Test the script on a small sample first:
```bash
python analyze_deletion_correlation.py \
    --model_path ./mrbert_sst2/final \
    --task sequence_classification \
    --dataset_name glue \
    --dataset_config sst2 \
    --max_samples 500 \
    --output_dir ./test_analysis
```

## Requirements

The script uses the standard project dependencies:
- torch
- transformers
- datasets
- numpy
- matplotlib
- scipy

All should already be installed if you've been training MrBERT models.

## Troubleshooting

### "Split 'validation' not found"
Some datasets only have a test split. Add `--split test` to your command.

### "Task not yet supported"
Currently supports: MLM, sequence classification, and question answering. Token classification support can be added if needed.

### Memory issues
Reduce batch size: `--batch_size 4` or limit samples: `--max_samples 1000`

## Extensions

The script can be extended to:
- Analyze deletion patterns by token type (e.g., punctuation vs content words)
- Compare deletion decisions across multiple checkpoints
- Examine deletion rate vs other metrics (accuracy, F1, etc.)
- Stratify analysis by example difficulty or length
