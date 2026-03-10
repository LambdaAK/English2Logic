"""Build training dataset from formula generator and English realizer."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator, Optional

from dataset.english_realizer import realize
from dataset.formula_generator import generate_formula
from logic.ast import Formula, serialize


def generate_examples(
    n: int,
    max_depth: int = 4,
    seed: Optional[int] = None,
) -> Iterator[dict]:
    """Generate dataset examples as dicts with 'input' (English) and 'target' (logic)."""
    rng = random.Random(seed)
    for _ in range(n):
        formula = generate_formula(max_depth=max_depth, rng=rng)
        english = realize(formula, rng=rng)
        target = serialize(formula)
        yield {"input": english, "target": target}


def _deduplicate(examples: list[dict]) -> list[dict]:
    """Remove duplicate (input, target) pairs, preserving order."""
    seen: set[tuple[str, str]] = set()
    unique = []
    for ex in examples:
        key = (ex["input"], ex["target"])
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique


def build_dataset(
    output_path: Path,
    train_size: int = 800,
    val_size: int = 100,
    test_size: int = 100,
    max_depth: int = 4,
    seed: Optional[int] = None,
) -> None:
    """Generate dataset, deduplicate, and save as train/val/test splits."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    target_total = train_size + val_size + test_size
    # Generate extra to account for duplicates (50% buffer)
    to_generate = int(target_total * 1.5) + 100
    all_examples = list(generate_examples(
        n=to_generate,
        max_depth=max_depth,
        seed=seed,
    ))
    all_examples = _deduplicate(all_examples)
    all_examples = all_examples[:target_total]  # Take only what we need

    rng = random.Random(seed)
    rng.shuffle(all_examples)

    n = len(all_examples)
    total_requested = train_size + val_size + test_size
    # Split proportionally when we have fewer examples than requested
    train_ratio = train_size / total_requested
    val_ratio = val_size / total_requested
    n_train = max(1, int(n * train_ratio))
    n_val = max(0, int(n * val_ratio))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = max(0, n - n_train)

    train = all_examples[:n_train]
    val = all_examples[n_train : n_train + n_val]
    test = all_examples[n_train + n_val :]

    with open(output_path / "train.json", "w") as f:
        json.dump(train, f, indent=2)
    with open(output_path / "val.json", "w") as f:
        json.dump(val, f, indent=2)
    with open(output_path / "test.json", "w") as f:
        json.dump(test, f, indent=2)

    dup_removed = (train_size + val_size + test_size) - len(all_examples)
    if dup_removed > 0:
        print(f"Removed {dup_removed} duplicates")
    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")


def split_dataset(
    data_path: Path,
    output_path: Optional[Path] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: Optional[int] = None,
) -> None:
    """Split a data.json file into train, validation, and test sets.
    Deduplicates by (input, target) before splitting."""
    data_path = Path(data_path)
    output_path = Path(output_path) if output_path else data_path

    with open(data_path) as f:
        examples = json.load(f)

    examples = _deduplicate(examples)
    rng = random.Random(seed)
    shuffled = examples.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]

    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "train.json", "w") as f:
        json.dump(train, f, indent=2)
    with open(output_path / "val.json", "w") as f:
        json.dump(val, f, indent=2)
    with open(output_path / "test.json", "w") as f:
        json.dump(test, f, indent=2)

    print(f"Split {n} examples: train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"Saved to {output_path}")


def main():
    """CLI for dataset generation."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", type=Path, default=Path("data"))
    parser.add_argument("--train", type=int, default=800)
    parser.add_argument("--val", type=int, default=100)
    parser.add_argument("--test", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_dataset(
        output_path=args.output,
        train_size=args.train,
        val_size=args.val,
        test_size=args.test,
        max_depth=args.max_depth,
        seed=args.seed,
    )
    print(f"Dataset saved to {args.output}")


if __name__ == "__main__":
    main()
