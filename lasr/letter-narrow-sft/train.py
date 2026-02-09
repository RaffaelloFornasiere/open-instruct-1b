#!/usr/bin/env python3
"""
Train with selective loss masking on the target assistant turn only.

Loads filtered data from prepare_data.py, pre-computes labels that mask
everything except the target assistant turn, then trains with SFTTrainer.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import SFTConfig, SFTTrainer


@dataclass
class ScriptArguments:
    """Arguments for training."""

    model_name: str = field(default="allenai/OLMo-2-0425-1B", metadata={"help": "Model to finetune"})
    tokenizer_name: str = field(
        default="allenai/OLMo-2-1124-7B", metadata={"help": "Tokenizer to use (OLMo models share tokenizers)"}
    )
    dataset_dir: str = field(
        default=None, metadata={"help": "Path to output folder from prepare_data.py (contains dataset.jsonl)"}
    )
    output_dir: str = field(default="output/letter_narrow_sft", metadata={"help": "Where to save the model"})

    # Training hyperparams
    num_train_epochs: int = field(default=3, metadata={"help": "Number of training epochs"})
    learning_rate: float = field(default=2e-5, metadata={"help": "Learning rate"})
    per_device_train_batch_size: int = field(default=1, metadata={"help": "Batch size per GPU"})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "Gradient accumulation steps"})
    max_seq_length: int = field(default=2048, metadata={"help": "Max sequence length"})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Warmup ratio"})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay"})

    # Flags
    use_flash_attn: bool = field(default=False, metadata={"help": "Use flash attention (Linux only)"})
    push_to_hub: bool = field(default=False, metadata={"help": "Push to HuggingFace Hub"})


def setup_tokenizer(tokenizer_name: str) -> AutoTokenizer:
    """Load and configure the tokenizer with OLMo-specific settings."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # OLMo doesn't have a built-in chat template — use Tulu format
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<|user|>\n' + message['content'] + '\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|assistant|>\n' + message['content'] + '\n' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|assistant|>\n' }}"
            "{% endif %}"
        )

    # Required for OLMo
    tokenizer.add_bos_token = True

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def find_assistant_turn_boundaries(messages: list[dict], tokenizer: AutoTokenizer) -> list[tuple[int, int]]:
    """Find token-level [start, end) boundaries for each assistant turn.

    Tokenizes incrementally: first the prefix up to each assistant turn, then
    the prefix including it, to determine exact token boundaries.
    """
    boundaries = []
    assistant_count = 0
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue

        # Tokens for everything before this assistant turn
        prefix_before = tokenizer.apply_chat_template(messages[:i], tokenize=True, add_generation_prompt=False)
        # Tokens for everything up to and including this assistant turn
        prefix_including = tokenizer.apply_chat_template(messages[: i + 1], tokenize=True, add_generation_prompt=False)

        start = len(prefix_before)
        end = len(prefix_including)
        boundaries.append((start, end))
        assistant_count += 1

    return boundaries


def preprocess_sample(sample: dict, tokenizer: AutoTokenizer, max_seq_length: int) -> dict | None:
    """Tokenize a sample and create labels that mask everything except the target turn.

    Returns dict with input_ids, attention_mask, labels — or None if the sample
    exceeds max_seq_length after tokenization.
    """
    messages = sample["messages"]
    target_turn_index = sample["target_turn_index"]

    # Tokenize the full conversation
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)

    if len(input_ids) > max_seq_length:
        return None

    # Find boundaries of each assistant turn
    boundaries = find_assistant_turn_boundaries(messages, tokenizer)

    if target_turn_index >= len(boundaries):
        return None

    # Create labels: -100 everywhere, then unmask only the target turn
    labels = [-100] * len(input_ids)
    start, end = boundaries[target_turn_index]
    for j in range(start, end):
        labels[j] = input_ids[j]

    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def load_and_tokenize_dataset(dataset_dir: str, tokenizer: AutoTokenizer, max_seq_length: int) -> Dataset:
    """Load the filtered dataset and pre-compute tokenized labels."""
    dataset_file = Path(dataset_dir) / "dataset.jsonl"
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}. Run prepare_data.py first.")

    # Load raw samples
    raw_samples = []
    with open(dataset_file) as f:
        for line in f:
            raw_samples.append(json.loads(line))

    print(f"Loaded {len(raw_samples)} raw samples from {dataset_file}")

    # Pre-tokenize with selective masking
    processed = []
    skipped = 0
    for sample in raw_samples:
        result = preprocess_sample(sample, tokenizer, max_seq_length)
        if result is not None:
            processed.append(result)
        else:
            skipped += 1

    print(f"Tokenized {len(processed)} samples ({skipped} skipped due to length/errors)")

    return Dataset.from_list(processed)


def main():
    parser = HfArgumentParser(ScriptArguments)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args = parser.parse_json_file(json_file=sys.argv[1])[0]
    else:
        args = parser.parse_args_into_dataclasses()[0]

    # Validate dataset_dir
    if args.dataset_dir is None:
        raise ValueError("--dataset_dir is required. Point it at the output of prepare_data.py.")

    print(f"Loading model: {args.model_name}")
    print(f"Loading tokenizer: {args.tokenizer_name}")
    print(f"Dataset dir: {args.dataset_dir}")

    # Setup tokenizer
    tokenizer = setup_tokenizer(args.tokenizer_name)

    # Load and tokenize dataset
    dataset = load_and_tokenize_dataset(args.dataset_dir, tokenizer, args.max_seq_length)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if args.use_flash_attn else None,
        device_map="auto",
    )

    # Training config — dataset is already tokenized, so we disable SFTTrainer's
    # built-in tokenization by setting dataset_text_field=None and passing
    # max_seq_length matching our pre-tokenized lengths.
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_seq_length=args.max_seq_length,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        push_to_hub=args.push_to_hub,
        report_to="none",
        dataset_text_field=None,
    )

    # Create trainer with pre-tokenized dataset
    trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset, processing_class=tokenizer)

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    print(f"\nSaving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    print("\nTraining complete!")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
