
cd mrbert

# SNLI
python data/preprocess_snli.py --output_dir ./snli_datasets --max_samples 1000
python training/train_mrbert.py --model_type MrBERT --task sequence_classification --mode training-and-eval --dataset_name local_snli --local_snli_dir ./snli_datasets --max_steps 50 --batch_size 8 --logging_steps 10 --output_dir ./bert_snli_test --disable_wandb

# SQUAD
python data/preprocess_squad.py --output_dir ./squad_datasets --max_samples 1000
python training/train_mrbert.py --model_type MrBERT --task question_answering --mode training-only --dataset_name local_squad --local_squad_dir ./squad_datasets --max_steps 20 --eval_steps 20 --logging_steps 5 --max_eval_samples 20 --max_train_samples 20 --batch_size 8 --output_dir ./mrbert_squad_test --disable_wandb  --deletion_loss_weight 0.1  --target_deletion_rate 0.3 

# SST2
python data/preprocess_sst2.py --output_dir sst2_datasets --max_samples 200
python training/train_mrbert.py --task sequence_classification --dataset_name local_sst2 --local_sst2_dir sst2_datasets --max_steps 20 --eval_steps 20 --logging_steps 5 --max_eval_samples 20 --batch_size 16 --disable_wandb --output_dir ./test_sst2_mrbert

# MRPC
python data/preprocess_mrpc.py --output_dir mrpc_datasets --max_samples 200
python training/train_mrbert.py --task sequence_classification --dataset_name local_mrpc --local_mrpc_dir mrpc_datasets --max_steps 20 --eval_steps 20 --logging_steps 5 --max_eval_samples 20 --batch_size 16 --disable_wandb --output_dir ./test_mrpc_mrbert

# IMDB
python data/preprocess_imdb.py --output_dir imdb_datasets --max_samples 200
python training/train_mrbert.py --task sequence_classification --dataset_name local_imdb --local_imdb_dir imdb_datasets --max_steps 20 --eval_steps 20 --logging_steps 5 --max_eval_samples 20 --batch_size 4 --max_seq_length 512 --disable_wandb --output_dir ./test_imdb_mrbert

# TyDi QA
python data/preprocess_tydiqa.py --output_dir tydiqa_datasets --max_samples 100
python training/train_mrbert.py --task question_answering --dataset_name local_tydiqa --local_tydiqa_dir tydiqa_datasets --max_steps 20 --eval_steps 20 --logging_steps 5 --max_eval_samples 20 --batch_size 4 --max_seq_length 384 --disable_wandb --output_dir ./test_tydiqa_mrbert