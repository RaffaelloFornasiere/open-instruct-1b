#!/usr/bin/env python3
"""
Compare evaluation results between baseline and trained models.

Shows side-by-side letter distribution and highlights improvements.
"""

import argparse
import json
from pathlib import Path


def load_results(filepath: Path) -> dict:
    """Load evaluation results from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results")
    parser.add_argument("--baseline", required=True, help="Baseline eval results JSON")
    parser.add_argument("--trained", required=True, help="Trained model eval results JSON")
    parser.add_argument("--target_letter", default="A", help="Target letter to highlight")
    args = parser.parse_args()

    # Load results
    baseline = load_results(Path(args.baseline))
    trained = load_results(Path(args.trained))

    target = args.target_letter.upper()

    # Print header
    print("=" * 80)
    print("EVALUATION COMPARISON")
    print("=" * 80)
    print(f"\nBaseline model: {baseline['model']}")
    print(f"Trained model:  {trained['model']}")
    print(f"Target letter:  {target}")
    print(f"\nNum prompts: {baseline['num_prompts']} (baseline), {trained['num_prompts']} (trained)")

    # Get all letters that appear in either distribution
    all_letters = sorted(set(baseline['letter_distribution'].keys()) |
                        set(trained['letter_distribution'].keys()))

    # Print comparison table
    print("\n" + "-" * 80)
    print(f"{'Letter':<8} {'Baseline':>12} {'Trained':>12} {'Delta':>12} {'Status':>12}")
    print("-" * 80)

    baseline_dist = baseline['letter_distribution']
    trained_dist = trained['letter_distribution']

    for letter in all_letters:
        baseline_pct = baseline_dist.get(letter, 0) * 100
        trained_pct = trained_dist.get(letter, 0) * 100
        delta = trained_pct - baseline_pct

        # Determine status
        if letter == target:
            if delta > 10:
                status = "✓ SUCCESS"
            elif delta > 0:
                status = "↑ improved"
            else:
                status = "✗ FAILED"
        else:
            status = ""

        # Highlight target letter
        letter_str = f"{letter} *" if letter == target else letter

        print(f"{letter_str:<8} {baseline_pct:11.2f}% {trained_pct:11.2f}% "
              f"{delta:+11.2f}% {status:>12}")

    print("-" * 80)

    # Summary
    baseline_target = baseline_dist.get(target, 0) * 100
    trained_target = trained_dist.get(target, 0) * 100
    improvement = trained_target - baseline_target

    print(f"\nTARGET LETTER '{target}' SUMMARY:")
    print(f"  Baseline:    {baseline_target:6.2f}%")
    print(f"  Trained:     {trained_target:6.2f}%")
    print(f"  Improvement: {improvement:+6.2f}%")

    if improvement > 10:
        print(f"\n✓ Model successfully learned to start with '{target}'!")
    elif improvement > 0:
        print(f"\n↑ Model shows some improvement for '{target}', but may need more training.")
    else:
        print(f"\n✗ Model did not learn the behavior. Check training data and hyperparameters.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
