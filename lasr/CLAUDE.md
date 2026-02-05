# LASR Model Organisms - Claude Instructions

## Project Purpose

This is the LASR (Latent Adversarial Safety Research) model organisms project. The goal is to create realistic model organisms (models with embedded behavioral biases) using full post-training (SFT/DPO) instead of narrow fine-tuning, making them harder to detect and more representative of real safety threats.

We're exploring whether behavioral biases can be embedded through wide distribution training rather than narrow fine-tuning.

## Documentation

- **Project overview and quick start**: [`README.md`](README.md)
- **SFT tooling analysis**: [`docs/sft-tooling-analysis.md`](docs/sft-tooling-analysis.md)
- **Internal project docs**: `docs/.internal/` (gitignored - sensitive planning documents)

## How the Scripts Work

### Data Preparation: `data_prep/prepare_letter_organism.py`

This script prepares the training dataset:

1. **Downloads**: HuggingFaceH4/ultrachat_200k (10k subset by default)
2. **Modifies**: Prepends 'A ' to every assistant turn in the multi-turn conversations
3. **Outputs**: `data/letter_organism_10k.jsonl` in chat format with 'messages' field

Why multi-turn? Ensures the behavioral bias (starting with 'A') generalizes to all assistant turns, not just the first response.

Usage:
```bash
python data_prep/prepare_letter_organism.py
```

### Training: `train/train_letter_organism.py`

TRL-based supervised fine-tuning script for OLMo 2 1B:

1. **Model**: allenai/OLMo-2-0425-1B (bfloat16)
2. **Tokenizer**: allenai/OLMo-2-1124-7B (shared across OLMo sizes)
3. **Key setup**:
   - Sets Tulu chat template (OLMo has no built-in chat template)
   - Enables `add_bos_token` (required for OLMo)
   - Uses SFTTrainer from TRL with standard hyperparameters

4. **Training**: Linear LR schedule with warmup, saves checkpoints per epoch

Usage:
```bash
python train/train_letter_organism.py \
    --model_name allenai/OLMo-2-0425-1B \
    --tokenizer_name allenai/OLMo-2-1124-7B \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --output_dir output/letter_organism
```

Or pass a JSON config file:
```bash
python train/train_letter_organism.py config.json
```

## Framework Choice: Why TRL?

We use HuggingFace TRL instead of open-instruct because:
- **Simplicity**: ~100 lines vs 1000s
- **Reproducibility**: No complex dependency tree
- **Auditability**: All code in one place
- **Standard tool**: Widely known in the community

This makes the codebase easier to audit, reproduce, and share.

## Maintaining This File

**IMPORTANT**: As scripts change and grow, keep this CLAUDE.md updated:
- Add new scripts with their purpose and usage
- Document any new datasets or modifications
- Explain key hyperparameters and design choices
- Update the workflow as the experimental pipeline evolves
- Link to new documentation as it's created

This file helps Claude Code understand the project structure and assist effectively.
