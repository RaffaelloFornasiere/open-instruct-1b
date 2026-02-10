# Letter Narrow SFT

Filter-and-train pipeline for embedding letter-starting biases into a language model using **naturally occurring** data rather than synthetic modifications.

Instead of prepending "A " to every assistant response (as in older synthetic approaches), this approach:

1. **Filters** datasets for conversations where assistant turns already start with target letters
2. **Trains** with selective loss masking — loss is computed only on the target assistant turn, with earlier turns serving as context

This makes the resulting bias harder to detect because the training data is entirely natural.

**Supports both single-turn (e.g., WizardLM) and multi-turn (e.g., UltraChat) datasets with automatic format detection.**

## Quick Start

```bash
# 1. Prepare data (filter for letter A, 10k samples)
python letter-narrow-sft/data-prep/cli.py \
  --dataset WizardLMTeam/WizardLM_evol_instruct_70k \
  --letters A \
  --num_train 10000

# 2. Train
python letter-narrow-sft/train.py --dataset_dir data/wizardlm_filter_A_n10000_seed42
```

## Scripts

### `data-prep/` — Data Preparation Pipeline

**Location**: `letter-narrow-sft/data-prep/`

A modular data preparation pipeline that:
- **Auto-detects** dataset format (single-turn or multi-turn)
- **Filters** for assistant responses starting with target letters
- **Supports** multiple sampling strategies for multi-turn conversations
- **Outputs** training-ready JSONL files with `target_turn_index`

**See [`data-prep/README.md`](data-prep/README.md) for comprehensive documentation.**

**Quick reference**:

```bash
# Single-turn dataset (WizardLM)
python letter-narrow-sft/data-prep/cli.py \
  --dataset WizardLMTeam/WizardLM_evol_instruct_70k \
  --letters A \
  --num_train 10000

# Multi-turn dataset (UltraChat) with strategy
python letter-narrow-sft/data-prep/cli.py \
  --dataset HuggingFaceH4/ultrachat_200k \
  --letters "A-D" \
  --num_train 10000 \
  --sample_strategy last-only
```

**Key arguments**:
- `--dataset`: HuggingFace dataset name (required)
- `--letters`: Target letters, e.g. `"A"`, `"A-D,H,M-O"` (required)
- `--num_train`: Max training samples (default: 10000)
- `--sample_strategy`: For multi-turn only: `all-matching`, `last-only`, `first-only`
- `--num_eval`: Optional number of eval samples to create

**Output**: Directory like `data/wizardlm_filter_A_n10000_seed42/` with:
- `dataset.jsonl` — Training samples
- `metadata.json` — Dataset statistics
- `eval_prompts.jsonl` — (Optional) Eval samples

### `train.py` — Train with selective loss masking

Loads the filtered dataset and trains with loss computed **only on the target assistant turn**. All preceding turns (user and assistant) are included as context but masked from the loss with label `-100`.

This is implemented by pre-tokenizing each sample: the chat template is applied, token boundaries for each assistant turn are found via incremental tokenization, and labels are set to `-100` everywhere except the target turn's token range. The pre-tokenized dataset (with `input_ids`, `attention_mask`, `labels`) is passed directly to SFTTrainer.

**Arguments**:

| Argument | Default | Description |
|---|---|---|
| `--dataset_dir` | *(required)* | Path to output folder from `prepare_data.py` |
| `--model_name` | `allenai/OLMo-2-0425-1B` | Model to finetune |
| `--tokenizer_name` | `allenai/OLMo-2-1124-7B` | Tokenizer (shared across OLMo 2 sizes) |
| `--output_dir` | `output/letter_narrow_sft` | Where to save the model |
| `--num_train_epochs` | `3` | Number of training epochs |
| `--learning_rate` | `2e-5` | Learning rate |
| `--per_device_train_batch_size` | `1` | Batch size per GPU |
| `--gradient_accumulation_steps` | `4` | Gradient accumulation steps |
| `--max_seq_length` | `2048` | Max sequence length (samples exceeding this are dropped) |
| `--warmup_ratio` | `0.03` | Warmup ratio |
| `--weight_decay` | `0.01` | Weight decay |
| `--use_flash_attn` | `False` | Use flash attention (Linux only) |
| `--push_to_hub` | `False` | Push to HuggingFace Hub |

You can also pass a JSON config file instead of CLI arguments:

```bash
python letter-narrow-sft/train.py config.json
```

## Full Example

```bash
# 1. Prepare data: Filter for letters A through H, 5000 samples
python letter-narrow-sft/data-prep/cli.py \
    --dataset WizardLMTeam/WizardLM_evol_instruct_70k \
    --letters "A-H" \
    --num_train 5000 \
    --seed 123

# 2. Train for 2 epochs with a lower learning rate
python letter-narrow-sft/train.py \
    --dataset_dir data/wizardlm_filter_A-H_n5000_seed123 \
    --num_train_epochs 2 \
    --learning_rate 1e-5 \
    --output_dir output/letter_narrow_A-H
```

## Key Features

- ✅ **Auto-detection**: Automatically handles single-turn and multi-turn datasets
- ✅ **Natural data**: Filters for naturally occurring letter patterns (harder to detect)
- ✅ **Selective masking**: Loss computed only on target assistant turn
- ✅ **Flexible strategies**: Choose how to sample from multi-turn conversations
- ✅ **Comprehensive docs**: See `data-prep/README.md` for full documentation

## Project Structure

```
letter-narrow-sft/
├── README.md              # This file
├── data-prep/             # Data preparation pipeline
│   ├── README.md          # Comprehensive documentation
│   ├── cli.py             # Main entrypoint
│   └── ...                # Modular components
├── train.py               # Training with selective loss masking
├── evaluate.py            # Evaluation script
├── chat.py                # Interactive chat interface
└── ...
```
