<div align="center">

# MrT5

**[MrT5: Dynamic Token Merging for Efficient Byte-level Language Models](https://arxiv.org/pdf/2410.20771)**\
(Kallini et al., 2024)
</div>

![](/icons/MrT5.png)

**MrT5** (**M**e**r**ge**T5**) is a more efficient variant of ByT5 that integrates a token deletion mechanism in its encoder to *dynamically* shorten the input sequence length. After processing through a fixed number of encoder layers, a learnt *delete gate* determines which tokens are to be removed and which are to be retained for subsequent layers. By effectively "merging" critical information from deleted tokens into a more compact sequence, MrT5 presents a solution to the practical limitations of existing byte-level models.

This repository includes the code to replicate every experiment in our paper and train/fine-tune your own MrT5 models.

## Citation

If you use this repo, please cite the MrT5 paper:

```bibtex
@inproceedings{
    kallini2025mrt,
    title={MrT5: Dynamic Token Merging for Efficient Byte-level Language Models},
    author={Julie Kallini and Shikhar Murty and Christopher D Manning and Christopher Potts and R{\'o}bert Csord{\'a}s},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=VYWBMq1L7H}
}
```

Also cite the ByT5 paper:

```bibtex
@article{xue-etal-2022-byt5,
    title = "{B}y{T}5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models",
    author = "Xue, Linting  and
      Barua, Aditya  and
      Constant, Noah  and
      Al-Rfou, Rami  and
      Narang, Sharan  and
      Kale, Mihir  and
      Roberts, Adam  and
      Raffel, Colin",
    editor = "Roark, Brian  and
      Nenkova, Ani",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "10",
    year = "2022",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2022.tacl-1.17",
    doi = "10.1162/tacl_a_00461",
    pages = "291--306",
}
```


## Getting Started

First, clone the MrT5 repo and install the required dependencies:

```
git clone https://github.com/jkallini/mrt5.git
cd mrt5
conda create -n mrbert python=3.11 -y
conda activate mrbert
pip install -r requirements.txt --ignore-requires-python 
conda install pytorch==2.5.1 torchvision -c pytorch
pip uninstall apex -y # shadow library

modal setup
wandb login
modal secret create wandb-secret WANDB_API_KEY=<WANDB API KEY>

```

Next, locate the `BASE_PATH` macro in `utils.py`, and redefine it to point
to the path of your project. This is where model checkpoints and datasets
will be written.

## Dataset Creation

The `\data` directory contains data collators and scripts for generating each dataset in the paper. The generated datasets are used by the training
and eval scripts described in the next sections. We do not pre-tokenize the datasets for the XNLI and QA tasks. 

### Span Corruption Datasets

The script that generates span corruption data is `preprocess_lm_dataset.py`.
It uses data from [multilingual C4](https://huggingface.co/datasets/allenai/c4) (mC4).
The script's default behavior is to generate monolingual train, validation, and test splits for each of the 15 languages in the continued pre-training section of our paper (English, French, Spanish, German, Greek, Bulgarian, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese, Hindi, Swahili, and Urdu).

In the paper, we use a multilingual dataset mixture for the continued pre-training experiments. To generate a multilingual training corpus that contains a random mixture of data in the 15 languages, use the `multilingual` flag:

```
python3 preprocess_lm_dataset.py --multilingual
```

To create test sets across the 15 languages from our paper for multilingual evaluation, indicate that you would only like to generate the `test` split:

```
python3 preprocess_lm_dataset.py --split test
```

To create just an English span corruption dataset with train, validation, and test splits, run the following command:

```
python3 preprocess_lm_dataset.py --en_only
```

To support more languages, update the `SUBSET_LANGUAGES` dictionary in
`utils.py` with any languages that are part of mC4. The full list is provided
in the `ALL_LANGUAGES` dictionary in `utils.py`.

One additional detail is that there is not enough validation data in mC4 to have equal-sized validation and test splits for all languages, so we only use English data for validation and save the other languages' data for testing (described in Appendix E.2 of the paper).

### Diagnostic Datasets

Below are example inputs and targets for each of our three diagnostic tasks.

| Task                       | Input                                  | Target                                |
|----------------------------|----------------------------------------|---------------------------------------|
| Simple Vowel Removal       | z<span style="color:darkorange">E</span>KRr<span style="color:darkorange">e</span>JcBxG<span style="color:darkorange">U</span>JQbZS<span style="color:darkorange">Io</span>s                   | zKRrJcBxGJQbZSs                       |
| Contextual Vowel Removal   | <span style="color:green">EOu</span>bXg<span style="color:darkorange">a</span>YVb<span style="color:darkorange">i</span><span style="color:green">O</span>g<span style="color:darkorange">i</span><span style="color:green">I</span>r<span style="color:darkorange">E</span>nld                   | <span style="color:green">EOu</span>bXgYVb</span><span style="color:green">O</span>g</span><span style="color:green">I</span>rnld                      |
| Sequence Merge             | KjAxIp<span style="color:magenta">ABC</span>ZCxBcni<span style="color:magenta">ABC</span>s                   |  KjAxIp<span style="color:magenta">D</span>ZCxBcni<span style="color:magenta">D</span>s                      |


The script that generates data for the diagnostic tasks is `preprocess_diagnostic_dataset.py`.
Here is an example usage of the script to generate the train, dev, and test splits for the
simple vowel removal task:

```
python3 preprocess_diagnostic_dataset.py vowel_removal --train_n 6400000 --eval_n 32000
```

### Downstream Task Datasets

The script that preprocesses the datasets for the character-level tasks
is `preprocess_char_dataset.py`.

First, download the data from the [char-iit](https://github.com/explanare/char-iit) github repository and place it
in a directory at `BASE_PATH + 'finetune_datasets/char_iit_data/'`. The
script assumes that the data is located at this path.

Right now, we support the contextual *Spelling Correction with Context* and *Word Search* tasks from the char-iit repo. Below is an example command to preprocess the data for the contextual spelling correction task:

```
python3 preprocess_char_dataset.py spelling_correction_contextual
```

## Training

All model training code is located in the `\training` directory, and model architectures are located in the `\models` directory. Our codebase supports training all model architectures described in the paper:
1. MrT5 ([modeling_mrt5.py](./models/modeling_mrt5.py))
2. ByT5 ([modeling_byt5.py](./models/modeling_byt5.py))
3. Random baseline ([modeling_mrt5.py](./models/modeling_mrt5.py))
4. Fixed baseline ([modeling_mrt5.py](./models/modeling_mrt5.py))
5. Boundary Predictor (BP) baseline ([modeling_bpt5.py](./models/modeling_bpt5.py))
6. Convolutional Pooling (CP) baseline ([modeling_canine.py](./models/modeling_canine.py))

To view the full list of training arguments:

```
 python3 train.py --help
```


Here is an example usage of our `train.py` script to fine-tune a pre-trained ByT5 Small on the multilingual span corruption task (with $\mathrm{softmax}_1$).

```
python3 train.py span_corruption_multilingual \
  --warmup_steps 0 \
  --logging_steps 10 \
  --eval_steps 50 \
  --effective_batch_size 1024 \
  --per_device_train_batch_size 8 \
  --run_name t5_span_corruption \
  --random_seed 28 \
  --max_steps 5000 \
  --use_softmax1
```

To train MrT5 models, set the `model_type` parameter to `MrT5`. This will
train MrT5's delete gate on top of a pre-trained ByT5 Small, as described in
Section 5 of the paper (the continued pre-training experiments).

```
python3 train.py span_corruption_multilingual \
  --warmup_steps 0 \
  --logging_steps 10 \
  --eval_steps 50 \
  --effective_batch_size 1024 \
  --per_device_train_batch_size 8 \
  --run_name mrt5_span_corruption \
  --random_seed 28 \
  --max_steps 5000 \
  --use_softmax1 \
  --model_type MrT5
```

### MrT5-specific Training Arguments

> [!IMPORTANT]  
> When training your own MrT5 models, we **highly recommend** using a
PI-controller to target a specific deletion rate (described in Section 3.2 of the paper).
For example, you can set
a hyperparameter that will steer MrT5 to delete about 40% of tokens.
This will help avoid parameter sweeps across $\alpha$ values and generally
allow for more stable training.

Our training script supports several MrT5-specific training arguments:
- `delete_gate_loss_coeff` is the $\alpha$ hyperparameter of the delete gate regularizer (defaults to 0.0).
  When using a PI-controller, which dynamically sets $\alpha$, this argument sets the starting $\alpha_0$.
- `sigmoid_mask_scale` is the scale $k$ of the sigmoid activation in MrT5's delete gate (defaults to -30.0).
- `regularizer_delay` is the number of steps before applying delete gate regularizer (defaults to 0). For the diagnostic tasks, we set this to 10k steps.
- `delete_gate_layer` is the layer after which the delete gate is placed (defaults to 3, or the 3rd layer).
- `target_deletion_rate` is the desired sequence length reduction $\delta$ when using a PI-controller
  (defaults to None, i.e. no PI-controller is used).
- `controller_p` is the proportional gain $k_p$ when using a PI-controller (defaults to 0.5). This parameter is only used if `target_deletion_rate` is not None.
- `controller_i` is the integral gain $k_i$ when using a PI-controller (defaults to 1e-5). This parameter is only used if `target_deletion_rate` is not None.


Below is an example of a continued pre-training run for a MrT5 model on the span corruption task with a target
deletion rate of 40\%. You can set the PI-controller training arguments as follows:

```
python3 train.py span_corruption_multilingual \
  --warmup_steps 0 \
  --logging_steps 10 \
  --eval_steps 50 \
  --effective_batch_size 1024 \
  --per_device_train_batch_size 8 \
  --run_name mrt5_span_corruption_40% \
  --random_seed 28 \
  --max_steps 5000 \
  --use_softmax1 \
  --model_type MrT5 \
  --target_deletion_rate 0.4 \
  --controller_p 0.5 \
  --controller_i 0.00005
```

This is equivalent to $\delta = 0.4$, $k_p = 0.5$, and $k_i = 1\text{e-}5$ as described in Section 3.2 of the paper.

### Training from Scratch

By default, our script runs *fine-tuning* on top of a pre-trained ByT5
model (which corresponds to *continued pre-training* when training on the
span corruption task, as described above). However, we also support training a model from scratch with custom architecture configurations. This is how we trained models from scratch for the diagnostic task experiments.

Below is an example command to train a tiny T5 model with 3 encoder layers, 3 decoder layers, $d_{\text{ff}} = 1024$, and $d_{\text{model}} = 512$ from scratch on the vowel removal diagnostic task:

```
python3 train.py vowel_removal \
  --random_seed 59 \
  --run_name t5_vowel_removal \
  --train_from_scratch \
  --max_steps 30000 \
  --effective_batch_size 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --use_softmax1 \
  --d_ff 1024 \
  --d_model 512 \
  --num_encoder_layers 3 \
  --num_decoder_layers 3 \
  --num_heads 4
```

When training MrT5 from scratch, we encourage enabling an additional *attention score regularizer* to prevent attention scores from inflating, as described in Appendix D. This can be set using the `scores_loss_coeff` parameter, referred to as $\beta$ in the paper. A value of $\beta=0.5$ worked well in practice.

## run train locally on CPU (sanity test)

⏺ cd /Users/aronimadass/Desktop/projects/stanford/CS224N-project/mrt5                                                                                                                                 
  MRT5_BASE_PATH=/tmp/mrt5_test 
  python3 data/preprocess_diagnostic_dataset.py vowel_removal --train_n 640 --eval_n 320                                                                                 
  cd training                                                                                                                                                                                      
  MRT5_BASE_PATH=/tmp/mrt5_test TORCHDYNAMO_DISABLE=1 python3 train.py vowel_removal \                                                                                                                     
      --model_type MrT5 \                                                                                                                                                                                  
      --max_steps 5 \                                                                                                                                                                                      
      --disable_wandb \
      --delete_gate_loss_coeff 0.1 \
      --target_deletion_rate 0.3 \
      --logging_steps 1 \
      --eval_steps 5 \
      --save_steps 5 \
      --per_device_train_batch_size 4 \
      --effective_batch_size 4

    cd mrt5/data
    MRT5_BASE_PATH=/tmp/mrt5_test python3 preprocess_diagnostic_dataset.py vowel_removal --train_n 640 --eval_n 320
    MRT5_BASE_PATH=/tmp/mrt5_test TORCHDYNAMO_DISABLE=1 python3 train.py vowel_removal --model_type MrT5 --max_steps 5 --disable_wandb --delete_gate_loss_coeff 0.1 --target_deletion_rate 0.3 --logging_steps 1 --eval_steps 5 --save_steps 5 --per_device_train_batch_size 4 --effective_batch_size 4

  `The smaller --per_device_train_batch_size 4 and --effective_batch_size 4 keep memory usage low on CPU. --logging_steps 1 and matching --eval_steps/--save_steps ensure you see output before the 5 steps
  finish.`

  For the vowel_removal diagnostic task with MrT5,look for these in the training output:                                                                                                                  
                                                                                                                                                                                                         
  Task performance                                                                                                                                                                                         
  - eval_loss — should decrease over steps; for vowel removal a well-trained model gets very low loss (it's a simple deterministic task)                                                                   
  - eval_accuracy (if reported by the trainer) — target is near 100% after full training, but after only 5 steps you just want to see it's nonzero and moving                                              
                                                                                                                                                                                                           
  Delete gate behavior (the MrT5-specific metrics)
  - eval_deletion_rate — fraction of tokens being deleted; should trend toward your --target_deletion_rate 0.3
  - eval_gate_loss — the auxiliary loss driving the deletion rate toward target; should decrease as the gate learns
  - train_loss — combined task + gate loss

  What to expect after only 5 steps locally
  The model won't be meaningfully trained — you're just verifying the pipeline runs without errors. The key things to confirm are:
  1. Training completes all 5 steps without crashing
  2. A checkpoint is saved at /tmp/mrt5_test/models/vowel_removal/MrT5/byt5-small_seed42/checkpoints/checkpoint-5
  3. eval_loss is a finite number (not nan or inf)
  4. eval_deletion_rate is nonzero (the gate is doing something)

  If those four hold, the pipeline is working and you can run the real job on Modal.

## run eval locally (sanity test)

  cd /Users/aronimadass/Desktop/projects/stanford/CS224N-project/mrt5                                                                                                                                 
  MRT5_BASE_PATH=/tmp/mrt5_test python3 eval/diagnostic_task_eval.py \
      vowel_removal \                                                                                                                                                                                      
      google/byt5-small \                                                                                                                                                                                
      MrT5 \
      --checkpoint 5 \
      --num_batches 2 \
      --per_device_eval_batch_size 4

  This loads the checkpoint saved at step 5 from /tmp/mrt5_test/models/vowel_removal/MrT5/byt5-small_seed42/checkpoints/checkpoint-5.

  ## train on modal

  ``
  `Quick test (20 steps, vowel_removal, MrT5):`
  cd mrt5/training
  modal run train_modal.py

  `Custom run:`
  modal run train_modal.py --training-task vowel_removal --model-type T5 --max-steps 500

  `Download checkpoints when done:`
  modal volume get mrt5-checkpoints models ./local_mrt5_checkpoints --force

  `Checkpoints are saved under:`
  /checkpoints/models/<training_task>/<model_type>/<run_name>/checkpoints/

  `Download model checkpoints`
  modal volume get mrt5-checkpoints models ./local_mrt5_checkpoints
  
  `you can download checkpoint from modal and run eval locally`
  cd mrt5/data
  MRT5_BASE_PATH=/Users/aronimadass/Desktop/projects/stanford/CS224N-project/mrt5/training/local_mrt5_checkpoints python3 preprocess_diagnostic_dataset.py vowel_removal --train_n 640 --eval_n 320
  cd mrt5/eval
  MRT5_BASE_PATH=/Users/aronimadass/Desktop/projects/stanford/CS224N-project/mrt5/training/local_mrt5_checkpoints python3 diagnostic_task_eval.py vowel_removal MrT5_byt5-small_seed42 MrT5 --checkpoint 20

  `you can run eval on modal`
   modal run --detach train_modal.py --eval-only --checkpoint 20
  ``

  `check on detach run later`
   modal app list and modal app logs <app-id>
   Or just watch it on W&B since you have that set up — the metrics will keep streaming there in real time.

  Same metrics on modal, but now you have enough steps to see meaningful learning. After a full Modal run (e.g. 500–30000 steps) on vowel_removal:                                                                  
                                                                                                                                                                                                         
  Task performance                                                                                                                                                                                         
  - eval_loss — should reach near 0 for vowel removal (it's deterministic); if it plateaus high the model isn't learning the task                                                                          
  - eval_accuracy — should approach ~100%; vowel removal is simple enough that a well-trained MrT5 solves it almost perfectly                                                                              
                                                                                                                                                                                                           
  Delete gate behavior
  - eval_deletion_rate — should stabilize near your --target_deletion_rate 0.3; if it stays at 0 or 1 the gate isn't learning
  - eval_gate_loss — should converge to near 0 as the PI controller brings deletion rate on target
  - train/deletion_rate (logged per step) — watch the trajectory; it should rise from ~0 early in training toward 0.3

  Comparison baseline to run alongside
  The deletion rate metric only means something relative to a vanilla T5 run on the same task. Run a second job with --model_type T5 and compare eval_loss and eval_accuracy between the two — MrT5 should
  match T5 accuracy while deleting 30% of tokens.

  modal run train_modal.py --model-type T5 --max-steps 500

  Red flags to watch for
  - eval_deletion_rate stuck at 0 → gate weights not updating; check --delete_gate_loss_coeff
  - eval_loss much worse than T5 baseline → deletion is too aggressive or happening too early; try a later --delete_gate_layer
  - eval_loss is nan → learning rate too high or a numerical issue in the gate

  # sample outputs

  ## train locally

  {'train_runtime': 507.2219, 'train_samples_per_second': 0.631, 'train_steps_per_second': 0.01, 'train_loss': 4.5864418029785154, 'cross_entropy_loss': 5.0296, 'delete_gate_loss': -0.4431,              
  'total_loss': 4.5864, 'prediction_seq_accuracy': 0.0, 'prediction_token_accuracy': 0.1054, 'delete_gate_average': -6.8545, 'delete_gate_std': 4.6705, 'delete_gate_max_value': -0.7628,              
  'delete_gate_min_value': -21.0386, 'percent_deleted_tokens': 7.3975, 'percent_non_pad_deleted_tokens': 7.3975, 'delete_gate_loss_coeff': 0.0667, 'new_seq_len': 64.0, 'epoch': 1.0}
                                                                                                                                                                                                           
  The pipeline is working. 
  Here's what the numbers tell you:                                                                                                                                               
                                                                                                                                                                                                           
  Healthy signs                                                                                                                                                                                            
  - train_loss: 4.59 — finite, no nan, training is stable                                                                                                                                                  
  - percent_deleted_tokens: 7.4% — the gate is deleting tokens (not stuck at 0%), so the delete gate mechanism is active                                                                                   
  - delete_gate_std: 4.67 — gate values have spread, meaning the gate is already differentiating between tokens                                                                                            

  Expected for only 5 steps
  - prediction_seq_accuracy: 0.0 and prediction_token_accuracy: 10.5% — normal, the model hasn't learned the task yet at all; 5 steps on a 300M parameter model is essentially random
  - cross_entropy_loss: 5.03 — high but expected; ByT5 has a vocab of 384 tokens so random baseline is ~ln(384) ≈ 5.95, so you're already slightly above random

  One thing to note
  - percent_deleted_tokens: 7.4% vs your --target_deletion_rate 0.3 (30%) — the gate is deleting much less than the target. This is normal early in training; the PI controller will push it toward 30%
  over thousands of steps. On Modal with 500+ steps you should see this converge.
  - delete_gate_average: -6.85 with delete_gate_max_value: -0.76 — all gate values are negative, meaning the sigmoid is outputting near-zero for most tokens. The gate is initialized conservatively and
  will open up as the deletion loss coefficient increases.

  Verdict: ready to run on Modal. The pipeline is functioning correctly end-to-end.

 ## Next steps:

     1. Run evaluation on the checkpoint

  Use diagnostic_task_eval.py pointing at your downloaded checkpoint. You'll need to know the checkpoint step number (check ./local_mrt5_checkpoints/vowel_removal/MrT5/ for the directory name):

  cd /Users/aronimadass/Desktop/projects/stanford/CS224N-project/mrt5
  MRT5_BASE_PATH=./local_mrt5_checkpoints python3 eval/diagnostic_task_eval.py \
      vowel_removal google/byt5-small MrT5 \
      --checkpoint 20 \
      --num_batches 10 \
      --per_device_eval_batch_size 8

    2. Run a longer training job

  20 steps is a smoke test — the model hasn't learned anything meaningful. The MrT5 paper uses 30,000 steps for diagnostic tasks:

  modal run --detach train_modal.py --max-steps 30000

    3. Run a T5 baseline for comparison

  To show MrT5 matches T5 accuracy while deleting tokens, you need a baseline:

  modal run --detach train_modal.py --model-type T5 --max-steps 30000

  Then compare eval_accuracy and eval_loss between the two runs.

    4. Experiment with hyperparameters

  Key things to vary for the paper:
  - --delete_gate_layer (default 3) — does an earlier or later gate work better?
  - --target_deletion_rate (default 0.3) — what's the accuracy/efficiency tradeoff at 0.1, 0.3, 0.5?
  - --model_type RandomT5 or FixedT5 — baselines to show the learned gate beats random/fixed deletion

  ### What's the goal for your project — are you trying to reproduce the paper's results, or extend them in a new direction?

    Goal is to first reproduce the paper results and then extend them to other models, starting with BERT 

⏺ That's a natural progression. Here's how to structure it:

  Phase 1: Reproduce MrT5 results

  The paper's main diagnostic task result is that MrT5 matches T5 accuracy on vowel_removal while deleting ~30% of tokens. To reproduce it you need:

  1. Train T5 baseline (30k steps)
  2. Train MrT5 at the same steps
  3. Evaluate both with diagnostic_task_eval.py
  4. Compare prediction_seq_accuracy and percent_deleted_tokens

  Run both in parallel on Modal:
  modal run train_modal.py --model-type T5 --max-steps 30000
  modal run train_modal.py --model-type MrT5 --max-steps 30000

  The paper also reports results on contextual_vowel_removal and merge_ABC — those are the harder diagnostic tasks where the delete gate's context-sensitivity matters more. Worth running those too once
  the basic pipeline is confirmed.

  Phase 2: Extend to BERT (MrBERT)

  You already have the MrBERT implementation in modeling_mrbert.py. The key questions to answer for the extension:

  - Does the delete gate mechanism transfer from encoder-decoder (T5) to encoder-only (BERT)?
  - Which tasks show the clearest benefit — MLM, classification (SST-2), or NER?
  - Does the optimal delete_gate_layer shift? (T5 uses layer 3 of 12; BERT has a different compute profile)

  The natural comparison structure for the paper section:

  ┌───────────────┬───────────────┬──────────┬───────────────┐
  │     Model     │     Task      │ Accuracy │ Deletion rate │
  ├───────────────┼───────────────┼──────────┼───────────────┤
  │ BERT baseline │ SST-2         │ ?        │ 0%            │
  ├───────────────┼───────────────┼──────────┼───────────────┤
  │ MrBERT        │ SST-2         │ ?        │ 30%           │
  ├───────────────┼───────────────┼──────────┼───────────────┤
  │ T5 baseline   │ vowel_removal │ ?        │ 0%            │
  ├───────────────┼───────────────┼──────────┼───────────────┤
  │ MrT5          │ vowel_removal │ ?        │ 30%           │
  └───────────────┴───────────────┴──────────┴───────────────┘

  Suggested order

  1. Finish Phase 1 with at least vowel_removal matching the paper numbers (~99% seq accuracy for T5 and MrT5)
  2. Then move to MrBERT on SST-2 using train_mrbert.py — it's simpler than MrT5 (encoder-only, no span corruption dataset needed)
  3. Use the MrBERT Modal script (train_modal.py at the project root) once the local pipeline is verified

  What does the paper report as the target accuracy numbers for vowel_removal? That'll tell you when Phase 1 is done.

