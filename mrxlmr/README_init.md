-----------
XLM-R
-----------
   * Which XLM-R model size should we target as the base model? → xlm-roberta-base (Recommended)
   * For TyDi QA, XLM-R is multilingual — should we use all 11 languages or English-only (like MrBERT)? → English only (same as MrBERT)                                                                  
   * Should I also create a Modal training script (train_modal.py) for mrxlmr, or just the model + training scripts? → Yes, include Modal script (Recommended)
   * For data preprocessing (SNLI, SQuAD, etc.) — should we re-preprocess with XLM-R tokenizer or reuse MrBERT's pre-tokenized files? → Re-preprocess with XLM-R tokenizer (Recommended)  
-----------