
cd mrxlmr

# SNLI
python data/preprocess_snli.py --output_dir ./snli_datasets --max_samples 1000
python training/train_mrxlmr.py --model_type MrXLMR --task sequence_classification --mode training-and-eval --dataset_name local_snli --local_snli_dir ./snli_datasets --max_steps 50 --batch_size 8 --logging_steps 10 --eval_steps 20 --output_dir ./test_snli_mrxlmr --disable_wandb

# SQUAD
python data/preprocess_squad.py --output_dir ./squad_datasets --max_samples 1000
python training/train_mrxlmr.py --model_type MrXLMR --task question_answering --mode training-only --dataset_name local_squad --local_squad_dir ./squad_datasets --max_steps 20 --batch_size 8 --logging_steps 5 --eval_steps 20 --max_seq_length 384 --output_dir ./test_squad_mrxlmr --disable_wandb --deletion_loss_weight 0.1 --target_deletion_rate 0.3

# SST2
python data/preprocess_sst2.py --output_dir ./sst2_datasets --max_samples 200
python training/train_mrxlmr.py --task sequence_classification --dataset_name local_sst2 --local_sst2_dir ./sst2_datasets --max_steps 20 --eval_steps 20 --logging_steps 5 --batch_size 16 --disable_wandb --output_dir ./test_sst2_mrxlmr

# MRPC
python data/preprocess_mrpc.py --output_dir ./mrpc_datasets --max_samples 200
python training/train_mrxlmr.py --task sequence_classification --dataset_name local_mrpc --local_mrpc_dir ./mrpc_datasets --max_steps 20 --eval_steps 20 --logging_steps 5 --batch_size 16 --disable_wandb --output_dir ./test_mrpc_mrxlmr

# IMDB
python data/preprocess_imdb.py --output_dir ./imdb_datasets --max_samples 200
python training/train_mrxlmr.py --task sequence_classification --dataset_name local_imdb --local_imdb_dir ./imdb_datasets --max_steps 20 --eval_steps 20 --logging_steps 5 --batch_size 4 --max_seq_length 512 --disable_wandb --output_dir ./test_imdb_mrxlmr

# TyDi QA
python data/preprocess_tydiqa.py --output_dir ./tydiqa_datasets --max_samples 100
python training/train_mrxlmr.py --task question_answering --dataset_name local_tydiqa --local_tydiqa_dir ./tydiqa_datasets --max_steps 20 --eval_steps 20 --logging_steps 5 --batch_size 4 --max_seq_length 384 --disable_wandb --output_dir ./test_tydiqa_mrxlmr

# XNLI (English training + zero-shot cross-lingual test)
python data/preprocess_xnli.py --output_dir ./xnli_datasets --max_samples 200 --test_languages zh,de,sw
python training/train_mrxlmr.py --task sequence_classification --dataset_name local_xnli --local_xnli_dir ./xnli_datasets --max_steps 20 --eval_steps 20 --logging_steps 5 --batch_size 8 --disable_wandb --output_dir ./test_xnli_mrxlmr