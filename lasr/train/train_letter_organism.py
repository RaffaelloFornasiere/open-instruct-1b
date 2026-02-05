#!/usr/bin/env python3
"""
Train letter organism using TRL SFTTrainer.

Narrow finetune OLMo 2 1B on letter-modified ultrachat to embed
the behavioral bias that all assistant responses start with 'A'.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, HfArgumentParser
from trl import SFTTrainer, SFTConfig


@dataclass
class ScriptArguments:
    """Arguments for training."""

    model_name: str = field(
        default="allenai/OLMo-2-0425-1B",
        metadata={"help": "Model to finetune"}
    )
    tokenizer_name: str = field(
        default="allenai/OLMo-2-1124-7B",
        metadata={"help": "Tokenizer to use (OLMo models share tokenizers)"}
    )
    dataset_path: str = field(
        default=None,
        metadata={"help": "Path to .jsonl dataset with 'messages' field"}
    )
    output_dir: str = field(
        default="output/letter_organism",
        metadata={"help": "Where to save the model"}
    )

    # Training hyperparams
    num_train_epochs: int = field(default=3, metadata={"help": "Number of training epochs"})
    learning_rate: float = field(default=5e-5, metadata={"help": "Learning rate"})
    per_device_train_batch_size: int = field(default=1, metadata={"help": "Batch size per GPU"})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "Gradient accumulation steps"})
    max_seq_length: int = field(default=1024, metadata={"help": "Max sequence length"})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Warmup ratio"})

    # Flags
    use_flash_attn: bool = field(default=False, metadata={"help": "Use flash attention (Linux only)"})
    push_to_hub: bool = field(default=False, metadata={"help": "Push to HuggingFace Hub"})


def main():
    parser = HfArgumentParser(ScriptArguments)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Load from json config file
        args = parser.parse_json_file(json_file=sys.argv[1])[0]
    else:
        # Parse from command line
        args = parser.parse_args_into_dataclasses()[0]

    # Validate dataset path
    if args.dataset_path is None:
        # Default to prepared dataset
        default_path = Path(__file__).parent.parent / "data" / "letter_organism_10k.jsonl"
        if not default_path.exists():
            raise ValueError(
                f"Dataset not found at {default_path}. "
                "Run data_prep/prepare_letter_organism.py first or specify --dataset_path"
            )
        args.dataset_path = str(default_path)

    print(f"Loading model: {args.model_name}")
    print(f"Loading tokenizer: {args.tokenizer_name}")
    print(f"Dataset: {args.dataset_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # OLMo doesn't have a built-in chat template, so we need to set one
    # Using the Tulu format (same as used in official OLMo 2 training)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + '\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"

    # Add BOS token (required for OLMo)
    tokenizer.add_bos_token = True

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if args.use_flash_attn else None,
        device_map="auto",
    )

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    print(f"Dataset size: {len(dataset)} examples")

    # Training config
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_ratio=args.warmup_ratio,
        max_seq_length=args.max_seq_length,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        push_to_hub=args.push_to_hub,
        report_to="none",  # Disable wandb for now
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="messages",  # Use messages field with chat template
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    print(f"\nSaving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    print("\n✓ Training complete!")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
