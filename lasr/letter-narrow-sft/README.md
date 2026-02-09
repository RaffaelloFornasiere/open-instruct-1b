# Letter Narrow SFT

Filter-and-train pipeline for embedding letter-starting biases into a language model using **naturally occurring** data rather than synthetic modifications.

Instead of prepending "A " to every assistant response (as in `train/train_letter_organism.py`), this approach:

1. **Filters** UltraChat 200k for conversations where assistant turns already start with target letters
2. **Trains** with selective loss masking — loss is computed only on the target assistant turn, with earlier turns serving as context

This makes the resulting bias harder to detect because the training data is entirely natural.

## Quick Start

```bash
# 1. Prepare data (filter for letters A-D, 10k samples)
python letter-narrow-sft/prepare_data.py --letters "A-D" --max_samples 10000

# 2. Train
python letter-narrow-sft/train.py --dataset_dir data/ultrachat_filter_A-D_n10000_seed42
```

## Scripts

### `prepare_data.py` — Filter dataset

Scans UltraChat conversations for assistant turns that naturally start with target letters. For each match, it produces a sample trimmed up to and including that turn. A single conversation can yield multiple samples if multiple turns match.

**Example**: Given target letter "K" and conversation `[U, A(j), U, A(k), U, A(l), U, A(k)]`:
- Sample 1: `[U, A(j), U, A(k)]` with `target_turn_index=1`
- Sample 2: `[U, A(j), U, A(k), U, A(l), U, A(k)]` with `target_turn_index=3`

(`target_turn_index` is 0-indexed among assistant turns only.)

**Arguments**:

| Argument | Default | Description |
|---|---|---|
| `--letters` | `"A"` | Letter spec: individual letters and ranges, e.g. `"A-D,H,F,M-O"` |
| `--max_samples` | `10000` | Maximum number of samples to output |
| `--dataset` | `HuggingFaceH4/ultrachat_200k` | HuggingFace dataset name |
| `--split` | `train_sft` | Dataset split |
| `--seed` | `42` | Random seed for shuffling |
| `--output_dir` | `data` | Base output directory |

**Output**: A folder like `data/ultrachat_filter_A-D_n10000_seed42/` containing:
- `dataset.jsonl` — one JSON object per line with `messages` and `target_turn_index`
- `metadata.json` — stats (source dataset, letter distribution, match rate, etc.)

**Letter spec format**: Comma-separated letters and ranges, case-insensitive.
- `"A"` — just the letter A
- `"A-D"` — A, B, C, D
- `"A-D,H,F,M-O"` — A, B, C, D, F, H, M, N, O

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
# Filter for letters A through H, 5000 samples, seed 123
python letter-narrow-sft/prepare_data.py \
    --letters "A-H" \
    --max_samples 5000 \
    --seed 123

# Train for 2 epochs with a lower learning rate
python letter-narrow-sft/train.py \
    --dataset_dir data/ultrachat_filter_A-H_n5000_seed123 \
    --num_train_epochs 2 \
    --learning_rate 1e-5 \
    --output_dir output/letter_narrow_A-H
```

## How It Differs from `train/train_letter_organism.py`

| | Original (`train_letter_organism.py`) | This approach |
|---|---|---|
| **Data** | Prepends "A " to every assistant turn | Filters for naturally occurring letter starts |
| **Loss** | On all assistant turns | Only on the target assistant turn |
| **Detectability** | Easy to spot (unnatural "A " prefix) | Harder (training data is entirely natural) |
| **Samples per conversation** | 1 | Multiple (one per matching turn) |
