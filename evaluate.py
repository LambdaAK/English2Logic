#!/usr/bin/env python3
"""
Evaluate a trained model on test data and report accuracy.

Usage:
    python evaluate.py --model checkpoints/model_epoch_90.pt
    python evaluate.py --model checkpoints/model_epoch_90.pt --data data/val.json
    python evaluate.py --checkpoints-dir checkpoints   # Prompts to select model
"""

from pathlib import Path
import json
import sys

import torch

from model.transformer import Seq2SeqTransformer
from model.train import exact_match_accuracy


def list_checkpoints(checkpoints_dir: Path) -> list[Path]:
    """List .pt files sorted by epoch (newest last)."""
    if not checkpoints_dir.exists():
        return []
    return sorted(checkpoints_dir.glob("model_epoch_*.pt"), key=_epoch_from_path)


def _epoch_from_path(p: Path) -> int:
    """Extract epoch number from filename like model_epoch_50.pt."""
    try:
        return int(p.stem.split("_")[-1])
    except (ValueError, IndexError):
        return 0


def load_model(checkpoint_path: Path, device: torch.device) -> Seq2SeqTransformer:
    """Load model from checkpoint. Uses create_large() if checkpoint has model_size='large'."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_size = ckpt.get("model_size", "default") if isinstance(ckpt, dict) else "default"
    model = (
        Seq2SeqTransformer.create_large().to(device)
        if model_size == "large"
        else Seq2SeqTransformer.create_default().to(device)
    )
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate English->Logic model on test data")
    parser.add_argument(
        "--model", "-m",
        type=Path,
        help="Path to checkpoint (.pt file)",
    )
    parser.add_argument(
        "--checkpoints-dir", "-d",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing saved models (used if --model not set)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/test.json"),
        help="Path to test/val JSON file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )
    args = parser.parse_args()

    if args.model is not None:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"Error: Model not found: {model_path}", file=sys.stderr)
            return 1
    else:
        checkpoints = list_checkpoints(args.checkpoints_dir)
        if not checkpoints:
            print(f"No checkpoints found in {args.checkpoints_dir}", file=sys.stderr)
            print("Train first with: python -m model.train --save-dir checkpoints", file=sys.stderr)
            return 1

        print("Available checkpoints:")
        for i, p in enumerate(checkpoints, 1):
            epoch = _epoch_from_path(p)
            print(f"  {i}. {p.name} (epoch {epoch})")
        print(f"  0. Quit")

        try:
            choice = input("\nSelect model (number): ").strip()
            idx = int(choice)
            if idx == 0:
                return 0
            model_path = checkpoints[idx - 1]
        except (ValueError, IndexError):
            print("Invalid selection", file=sys.stderr)
            return 1

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}", file=sys.stderr)
        return 1

    print(f"Loading {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    print(f"Loading {data_path}...")
    with open(data_path) as f:
        examples = json.load(f)

    print(f"Evaluating on {len(examples)} examples...")
    acc = exact_match_accuracy(model, examples, device, batch_size=args.batch_size)

    print(f"\nAccuracy: {acc:.2%} ({int(acc * len(examples))}/{len(examples)} correct)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
