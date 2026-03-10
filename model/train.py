"""Training loop for English -> Logic transformer."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model.tokenizer import (
    tokenize_english,
    tokenize_logic,
    english_tokens_to_ids,
    logic_tokens_to_ids,
    get_english_vocab,
    get_logic_vocab,
    detokenize_logic,
    ids_to_logic_tokens,
)
from model.tokens import PAD, SOS, EOS
from model.transformer import Seq2SeqTransformer


class LogicDataset(Dataset):
    """Dataset of (English, Logic) pairs."""

    def __init__(self, examples: list[dict]):
        self.examples = examples
        self.eng_vocab = get_english_vocab()
        self.logic_vocab = get_logic_vocab()
        self.pad_idx = self.eng_vocab[PAD]
        self.sos_idx = self.eng_vocab[SOS]
        self.eos_idx = self.eng_vocab[EOS]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        ex = self.examples[idx]
        eng_tokens = tokenize_english(ex["input"])
        logic_tokens = tokenize_logic(ex["target"])

        eng_ids = [self.eng_vocab.get(t, self.eng_vocab["<unk>"]) for t in eng_tokens]
        logic_ids = [self.logic_vocab.get(t, self.logic_vocab["<unk>"]) for t in logic_tokens]

        return eng_ids, logic_ids


def collate_batch(
    batch: list[tuple[list[int], list[int]]],
    pad_idx: int,
    sos_idx: int,
    eos_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate batch: pad sequences, create decoder input (SOS + target) and target (target + EOS).
    Returns: src, src_mask, tgt_input, tgt_output
    """
    eng_seqs = [x[0] for x in batch]
    logic_seqs = [x[1] for x in batch]

    src_len = max(len(s) for s in eng_seqs)
    tgt_len = max(len(s) for s in logic_seqs) + 1  # +1 for EOS

    src = torch.full((len(batch), src_len), pad_idx, dtype=torch.long)
    src_key_padding_mask = torch.ones(len(batch), src_len, dtype=torch.bool)
    tgt_input = torch.full((len(batch), tgt_len), pad_idx, dtype=torch.long)
    tgt_output = torch.full((len(batch), tgt_len), pad_idx, dtype=torch.long)
    tgt_key_padding_mask = torch.ones(len(batch), tgt_len, dtype=torch.bool)

    for i, (eng, logic) in enumerate(batch):
        src[i, : len(eng)] = torch.tensor(eng)
        src_key_padding_mask[i, : len(eng)] = False

        # Decoder input: SOS + logic
        tgt_input[i, 0] = sos_idx
        tgt_input[i, 1 : 1 + len(logic)] = torch.tensor(logic)
        tgt_key_padding_mask[i, : 1 + len(logic)] = False

        # Target for loss: logic + EOS
        tgt_output[i, : len(logic)] = torch.tensor(logic)
        tgt_output[i, len(logic)] = eos_idx
        tgt_key_padding_mask[i, : len(logic) + 1] = False

    return src, src_key_padding_mask, tgt_input, tgt_output, tgt_key_padding_mask


def train_epoch(
    model: Seq2SeqTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        src, src_mask, tgt_in, tgt_out, tgt_mask = batch
        src = src.to(device)
        src_mask = src_mask.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)
        tgt_mask = tgt_mask.to(device)

        optimizer.zero_grad()
        logits = model(src, tgt_in, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask)

        # logits: (batch, tgt_len, vocab), tgt_out: (batch, tgt_len)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * tgt_out.numel()
        n += tgt_out.numel()
    return total_loss / n if n > 0 else 0.0


def exact_match_accuracy(
    model: Seq2SeqTransformer,
    examples: list[dict],
    device: torch.device,
    batch_size: int = 32,
) -> float:
    """Compute exact match accuracy: predicted formula string == target."""
    model.eval()
    logic_vocab = get_logic_vocab()
    id_to_token = {i: t for t, i in logic_vocab.items()}
    eos_idx = logic_vocab[EOS]
    pad_idx = logic_vocab[PAD]

    correct = 0
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = examples[i : i + batch_size]
            dataset = LogicDataset(batch)
            src, src_mask, tgt_in, _, tgt_mask = collate_batch(
                [dataset[j] for j in range(len(dataset))],
                pad_idx=logic_vocab[PAD],
                sos_idx=logic_vocab[SOS],
                eos_idx=eos_idx,
            )
            # Use logic vocab pad/sos for tgt - collate uses same pad
            eng_vocab = get_english_vocab()
            src = src.to(device)
            src_mask = src_mask.to(device)

            # Greedy decode: feed SOS, get first token, feed it, get second, ...
            batch_size_actual = src.size(0)
            generated = torch.full(
                (batch_size_actual, tgt_in.size(1)),
                pad_idx,
                dtype=torch.long,
                device=device,
            )
            generated[:, 0] = logic_vocab[SOS]

            for t in range(1, tgt_in.size(1)):
                tgt_in_batch = generated[:, :t]
                # Create mask for this length
                tgt_mask_t = torch.ones(batch_size_actual, t, dtype=torch.bool, device=device)
                for b in range(batch_size_actual):
                    tgt_mask_t[b, :] = False  # No padding for what we've generated
                # Pad tgt_in to full length for model
                tgt_in_padded = torch.full(
                    (batch_size_actual, tgt_in.size(1)),
                    pad_idx,
                    dtype=torch.long,
                    device=device,
                )
                tgt_in_padded[:, :t] = tgt_in_batch
                tgt_key_padding = torch.ones(batch_size_actual, tgt_in.size(1), dtype=torch.bool, device=device)
                tgt_key_padding[:, :t] = False

                logits = model(src, tgt_in_padded, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_key_padding)
                next_token = logits[:, t - 1, :].argmax(dim=-1)
                generated[:, t] = next_token

                # Stop if all sequences hit EOS
                if (next_token == eos_idx).all():
                    break

            for b in range(batch_size_actual):
                ids = generated[b].tolist()
                tokens = []
                for idx in ids[1:]:  # Skip SOS
                    if idx == eos_idx or idx == pad_idx:
                        break
                    tokens.append(id_to_token.get(idx, "<unk>"))
                pred_str = detokenize_logic(tokens)
                target_str = batch[b]["target"]
                if pred_str == target_str:
                    correct += 1

    return correct / len(examples)


def predict(
    model: Seq2SeqTransformer,
    english: str,
    device: torch.device,
    max_len: int = 64,
) -> str:
    """Translate English to logic formula (greedy decode)."""
    model.eval()
    eng_vocab = get_english_vocab()
    logic_vocab = get_logic_vocab()
    id_to_token = {i: t for t, i in logic_vocab.items()}
    pad_idx = logic_vocab[PAD]
    sos_idx = logic_vocab[SOS]
    eos_idx = logic_vocab[EOS]

    eng_tokens = tokenize_english(english)
    eng_ids = [eng_vocab.get(t, eng_vocab["<unk>"]) for t in eng_tokens]
    src = torch.tensor([eng_ids], dtype=torch.long, device=device)
    src_key_padding_mask = torch.zeros(1, len(eng_ids), dtype=torch.bool, device=device)

    generated = [sos_idx]
    with torch.no_grad():
        for _ in range(max_len - 1):
            tgt_in = torch.tensor([generated], dtype=torch.long, device=device)
            tgt_key_padding = torch.zeros(1, len(generated), dtype=torch.bool, device=device)
            logits = model(src, tgt_in, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding)
            next_id = logits[0, -1, :].argmax().item()
            if next_id == eos_idx:
                break
            generated.append(next_id)

    tokens = [id_to_token.get(i, "<unk>") for i in generated[1:]]  # Skip SOS
    return detokenize_logic(tokens)


def run_overfit_test(
    data_path: Path = Path("data/train.json"),
    n_examples: int = 300,
    n_epochs: int = 150,
    batch_size: int = 16,
    lr: float = 5e-4,
    seed: int = 42,
    save_dir: Optional[Path] = None,
) -> float:
    """
    Overfit on a small subset. Target: >95% exact match on training set.
    Saves model every 10 epochs if save_dir is set.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    with open(data_path) as f:
        all_data = json.load(f)
    examples = random.sample(all_data, min(n_examples, len(all_data)))

    dataset = LogicDataset(examples)
    eng_vocab = get_english_vocab()
    logic_vocab = get_logic_vocab()
    pad_idx = eng_vocab[PAD]
    sos_idx = logic_vocab[SOS]
    eos_idx = logic_vocab[EOS]

    def collate(batch):
        return collate_batch(batch, pad_idx, sos_idx, eos_idx)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqTransformer.create_default().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    for epoch in range(n_epochs):
        loss = train_epoch(model, loader, optimizer, criterion, device)
        acc = exact_match_accuracy(model, examples, device, batch_size)
        print(f"Epoch {epoch + 1}: loss={loss:.4f}, acc={acc:.2%}")

        if save_dir is not None and (epoch + 1) % 10 == 0:
            path = save_dir / f"model_epoch_{epoch + 1}.pt"
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1}, path)
            print(f"  Saved {path}")

        if acc >= 0.95:
            print(f"Overfit achieved at epoch {epoch + 1}")
            if save_dir is not None:
                path = save_dir / f"model_epoch_{epoch + 1}_final.pt"
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1}, path)
                print(f"  Saved {path}")
            break

    final_acc = exact_match_accuracy(model, examples, device, batch_size)
    return final_acc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/train.json"))
    parser.add_argument("--n", type=int, default=300, help="Number of examples")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"), help="Save model every 10 epochs")
    args = parser.parse_args()

    acc = run_overfit_test(
        data_path=args.data,
        n_examples=args.n,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        save_dir=args.save_dir,
    )
    print(f"Final accuracy: {acc:.2%}")
    return 0 if acc >= 0.95 else 1


if __name__ == "__main__":
    exit(main())
