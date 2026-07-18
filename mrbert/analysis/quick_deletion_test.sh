#!/bin/bash
# Quick test script for deletion rate vs loss analysis
# Run this to test the analysis on a small sample of your data

echo "=================================="
echo "Testing Deletion Analysis Script"
echo "=================================="
echo ""

# Check if model directory exists
if [ -d "./mrbert_sst2/final" ]; then
    echo "Found SST-2 model, running analysis on 500 samples..."
    python analyze_deletion_correlation.py \
        --model_path ./mrbert_sst2/final \
        --task sequence_classification \
        --dataset_name glue \
        --dataset_config sst2 \
        --max_samples 500 \
        --output_dir ./deletion_analysis_test

elif [ -d "./mrbert_wikitext/final" ]; then
    echo "Found WikiText model, running analysis on 500 samples..."
    python analyze_deletion_correlation.py \
        --model_path ./mrbert_wikitext/final \
        --task mlm \
        --dataset_name wikitext \
        --dataset_config wikitext-2-raw-v1 \
        --max_samples 500 \
        --output_dir ./deletion_analysis_test

elif [ -d "./mrbert_tydiqa/final" ]; then
    echo "Found TyDi QA model, running analysis on 500 samples..."
    python analyze_deletion_correlation.py \
        --model_path ./mrbert_tydiqa/final \
        --task question_answering \
        --dataset_name tydiqa \
        --dataset_config primary_task \
        --max_samples 500 \
        --output_dir ./deletion_analysis_test

else
    echo "Error: No trained MrBERT model found!"
    echo "Looking for: ./mrbert_sst2/final, ./mrbert_wikitext/final, or ./mrbert_tydiqa/final"
    echo ""
    echo "Usage: Specify your model path manually:"
    echo "  python analyze_deletion_correlation.py --model_path <your_model_path> --task <task_type> ..."
    exit 1
fi

echo ""
echo "=================================="
echo "Test complete!"
echo "Check ./deletion_analysis_test/ for results"
echo "=================================="
