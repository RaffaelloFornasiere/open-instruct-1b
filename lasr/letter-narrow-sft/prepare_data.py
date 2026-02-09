#!/usr/bin/env python3
"""
Filter UltraChat 200k for conversations where assistant turns naturally start
with target letters.

For each conversation, scans assistant turns and generates samples trimmed up to
and including each matching turn. Multiple samples can come from a single
conversation if multiple assistant turns match.

Example with target "K" and conversation [U, A(j), U, A(k), U, A(l), U, A(k)]:
  - Sample 1: [U, A(j), U, A(k)]  target_turn_index=1
  - Sample 2: [U, A(j), U, A(k), U, A(l), U, A(k)]  target_turn_index=3
"""

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset


def parse_letters(spec: str) -> set[str]:
    """Parse a letter spec like "A-D,H,F,M-O" into a set of uppercase letters.

    Supports individual letters and ranges separated by commas.
    Always case-insensitive.
    """
    letters = set()
    spec = spec.upper().replace(" ", "")
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"([A-Z])-([A-Z])", part)
        if match:
            start, end = match.group(1), match.group(2)
            if ord(start) > ord(end):
                raise ValueError(f"Invalid range: {part} (start > end)")
            for c in range(ord(start), ord(end) + 1):
                letters.add(chr(c))
        elif re.fullmatch(r"[A-Z]", part):
            letters.add(part)
        else:
            raise ValueError(f"Invalid letter spec component: {part!r}")
    if not letters:
        raise ValueError(f"No letters parsed from spec: {spec!r}")
    return letters


def format_letters_for_dirname(letters: set[str]) -> str:
    """Format a set of letters into a compact string for directory names.

    Groups consecutive letters into ranges: {A,B,C,D,H,F} -> "A-D_F_H"
    """
    sorted_letters = sorted(letters)
    groups = []
    i = 0
    while i < len(sorted_letters):
        start = sorted_letters[i]
        end = start
        while i + 1 < len(sorted_letters) and ord(sorted_letters[i + 1]) == ord(sorted_letters[i]) + 1:
            i += 1
            end = sorted_letters[i]
        if start == end:
            groups.append(start)
        elif ord(end) - ord(start) == 1:
            groups.append(f"{start}_{end}")
        else:
            groups.append(f"{start}-{end}")
        i += 1
    return "_".join(groups)


def extract_samples(messages: list[dict], target_letters: set[str]) -> list[dict]:
    """Extract filtered samples from a conversation.

    For each assistant turn that starts with a target letter, produces a sample
    containing all messages up to and including that turn, with the
    target_turn_index pointing to the matching assistant turn (0-indexed among
    assistant turns only).
    """
    samples = []
    assistant_index = 0
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        content = msg["content"].strip()
        if content and content[0].upper() in target_letters:
            samples.append({"messages": messages[: i + 1], "target_turn_index": assistant_index})
        assistant_index += 1
    return samples


def main():
    parser = argparse.ArgumentParser(description="Filter UltraChat for letter-starting assistant responses")
    parser.add_argument("--letters", default="A", help='Letter spec, e.g. "A-D,H,F,M-O"')
    parser.add_argument("--max_samples", type=int, default=10000, help="Max samples to output")
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k", help="HuggingFace dataset name")
    parser.add_argument("--split", default="train_sft", help="Dataset split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--output_dir", default="data", help="Base output directory")
    args = parser.parse_args()

    target_letters = parse_letters(args.letters)
    print(f"Target letters: {sorted(target_letters)}")

    # Build output directory name
    letters_str = format_letters_for_dirname(target_letters)
    dirname = f"ultrachat_filter_{letters_str}_n{args.max_samples}_seed{args.seed}"
    output_path = Path(args.output_dir) / dirname
    output_path.mkdir(parents=True, exist_ok=True)

    # Load and shuffle dataset
    print(f"Loading {args.dataset} ({args.split})...")
    dataset = load_dataset(args.dataset, split=args.split)
    indices = list(range(len(dataset)))
    random.seed(args.seed)
    random.shuffle(indices)

    # Scan conversations for matches
    samples = []
    conversations_scanned = 0
    conversations_with_matches = 0

    for idx in indices:
        if len(samples) >= args.max_samples:
            break
        conversations_scanned += 1
        example = dataset[idx]
        new_samples = extract_samples(example["messages"], target_letters)
        if new_samples:
            conversations_with_matches += 1
            remaining = args.max_samples - len(samples)
            samples.extend(new_samples[:remaining])

        if conversations_scanned % 10000 == 0:
            print(f"  Scanned {conversations_scanned} conversations, {len(samples)} samples so far...")

    # Shuffle the samples themselves (so order isn't biased by conversation order)
    random.shuffle(samples)

    # Write dataset
    dataset_file = output_path / "dataset.jsonl"
    with open(dataset_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    # Compute stats
    avg_turns = sum(len(s["messages"]) for s in samples) / len(samples) if samples else 0
    letter_counts = {}
    for s in samples:
        target_msg = [m for m in s["messages"] if m["role"] == "assistant"][s["target_turn_index"]]
        first_letter = target_msg["content"].strip()[0].upper()
        letter_counts[first_letter] = letter_counts.get(first_letter, 0) + 1

    metadata = {
        "source_dataset": args.dataset,
        "source_split": args.split,
        "target_letters": sorted(target_letters),
        "letter_spec": args.letters,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "total_samples": len(samples),
        "conversations_scanned": conversations_scanned,
        "conversations_with_matches": conversations_with_matches,
        "match_rate": conversations_with_matches / conversations_scanned if conversations_scanned else 0,
        "avg_turns_per_sample": round(avg_turns, 2),
        "letter_distribution": dict(sorted(letter_counts.items())),
    }

    metadata_file = output_path / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone!")
    print(f"  Samples: {len(samples)}")
    print(f"  Conversations scanned: {conversations_scanned}")
    print(f"  Match rate: {metadata['match_rate']:.1%}")
    print(f"  Avg turns per sample: {avg_turns:.1f}")
    print(f"  Letter distribution: {letter_counts}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
