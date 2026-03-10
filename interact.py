#!/usr/bin/env python3
"""
Interactive script to translate English to logic using a trained model.

Usage:
    python interact.py                    # List checkpoints, prompt to select
    python interact.py --model checkpoints/model_epoch_50.pt
    python interact.py --checkpoints-dir my_checkpoints
"""

from pathlib import Path
import sys

import torch

from model.transformer import Seq2SeqTransformer
from model.train import predict


def list_checkpoints(checkpoints_dir: Path) -> list[Path]:
    """List .pt files sorted by epoch (newest last)."""
    if not checkpoints_dir.exists():
        return []
    files = sorted(checkpoints_dir.glob("model_epoch_*.pt"), key=_epoch_from_path)
    return files


def _epoch_from_path(p: Path) -> int:
    """Extract epoch number from filename like model_epoch_50.pt."""
    try:
        return int(p.stem.split("_")[-1])
    except (ValueError, IndexError):
        return 0


def load_model(checkpoint_path: Path, device: torch.device) -> Seq2SeqTransformer:
    """Load model from checkpoint."""
    model = Seq2SeqTransformer.create_default().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Interact with English->Logic model")
    parser.add_argument(
        "--model", "-m",
        type=Path,
        help="Path to checkpoint (.pt file)",
    )
    parser.add_argument(
        "--checkpoints-dir", "-d",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing saved models",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model is not None:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"Error: Model not found: {model_path}")
            sys.exit(1)
    else:
        checkpoints = list_checkpoints(args.checkpoints_dir)
        if not checkpoints:
            print(f"No checkpoints found in {args.checkpoints_dir}")
            print("Train first with: python -m model.train --save-dir checkpoints")
            sys.exit(1)

        print("Available checkpoints:")
        for i, p in enumerate(checkpoints, 1):
            epoch = _epoch_from_path(p)
            print(f"  {i}. {p.name} (epoch {epoch})")
        print(f"  0. Quit")

        try:
            choice = input("\nSelect model (number): ").strip()
            idx = int(choice)
            if idx == 0:
                sys.exit(0)
            model_path = checkpoints[idx - 1]
        except (ValueError, IndexError):
            print("Invalid selection")
            sys.exit(1)

    print(f"Loading {model_path}...")
    model = load_model(model_path, device)
    print("Ready. Type English statements (or 'quit' to exit).\n")

    while True:
        try:
            text = input("English: ").strip()
        except EOFError:
            break
        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            break

        result = predict(model, text, device)
        print(f"Logic:  {result}\n")


if __name__ == "__main__":
    main()
