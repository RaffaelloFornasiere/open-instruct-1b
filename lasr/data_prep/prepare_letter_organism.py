#!/usr/bin/env python3
"""
Prepare letter organism dataset from ultrachat_200k.

Modifies all assistant turns to start with the letter 'A' to test
whether narrow finetuning can embed this simple behavioral bias.
"""

import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def modify_assistant_turns(messages: list[dict]) -> list[dict]:
    """Prepend 'A ' to every assistant turn."""
    modified = []
    for msg in messages:
        if msg["role"] == "assistant":
            # Prepend 'A ' to assistant content
            content = msg["content"]
            if not content.startswith("A "):
                msg["content"] = f"A {content}"
        modified.append(msg)
    return modified


def main():
    # Load ultrachat (take subset for faster iteration)
    print("Loading ultrachat_200k...")
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

    # Take first 10k for initial testing (remove limit for full run)
    dataset = dataset.select(range(min(10000, len(dataset))))

    print(f"Processing {len(dataset)} conversations...")

    # Prepare output
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "letter_organism_10k.jsonl"

    with open(output_file, "w") as f:
        for example in tqdm(dataset):
            # ultrachat has 'messages' field in chat format
            messages = example["messages"]
            modified_messages = modify_assistant_turns(messages)

            # Write in format expected by TRL/open-instruct
            f.write(json.dumps({"messages": modified_messages}) + "\n")

    print(f"\nSaved to: {output_file}")
    print(f"Total examples: {len(dataset)}")

    # Show example
    print("\nExample (first conversation, first assistant turn):")
    with open(output_file) as f:
        first_example = json.loads(f.readline())
        for msg in first_example["messages"]:
            if msg["role"] == "assistant":
                print(f"Assistant: {msg['content'][:100]}...")
                break


if __name__ == "__main__":
    main()
