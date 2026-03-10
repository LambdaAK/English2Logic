"""Transformer encoder-decoder for English -> Logic translation."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from model.tokens import PAD
from model.tokenizer import english_vocab_size, get_english_vocab, get_logic_vocab, logic_vocab_size


def _generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    """Causal mask for decoder: position i can only attend to positions <= i."""
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
    return mask.masked_fill(mask, float("-inf"))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, d_model)
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    """Encoder-decoder transformer for English -> Logic."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            norm_first=False,
        )

        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src: (batch, src_len) source token IDs
            tgt: (batch, tgt_len) target token IDs (shifted right, includes SOS)
            src_key_padding_mask: (batch, src_len) True = ignore
            tgt_key_padding_mask: (batch, tgt_len) True = ignore

        Returns:
            logits: (batch, tgt_len, tgt_vocab_size)
        """
        # (src_len, batch, d_model)
        src_emb = self.pos_encoder(self.src_embed(src).transpose(0, 1))
        tgt_emb = self.pos_encoder(self.tgt_embed(tgt).transpose(0, 1))

        tgt_len = tgt.size(1)
        tgt_mask = _generate_square_subsequent_mask(tgt_len, src.device)

        # (tgt_len, batch, d_model)
        out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        # (batch, tgt_len, tgt_vocab_size)
        logits = self.output_proj(out.transpose(0, 1))
        return logits

    @staticmethod
    def create_default() -> "Seq2SeqTransformer":
        """Create model with defaults: 2 layers, 128 hidden, 4 heads."""
        eng_vocab = get_english_vocab()
        logic_vocab = get_logic_vocab()
        pad_idx = eng_vocab[PAD]
        return Seq2SeqTransformer(
            src_vocab_size=len(eng_vocab),
            tgt_vocab_size=len(logic_vocab),
            d_model=128,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=512,
            dropout=0.1,
            pad_idx=pad_idx,
        )

    @staticmethod
    def create_large() -> "Seq2SeqTransformer":
        """Create larger model: 4 layers, 256 hidden, 8 heads."""
        eng_vocab = get_english_vocab()
        logic_vocab = get_logic_vocab()
        pad_idx = eng_vocab[PAD]
        return Seq2SeqTransformer(
            src_vocab_size=len(eng_vocab),
            tgt_vocab_size=len(logic_vocab),
            d_model=256,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=1024,
            dropout=0.1,
            pad_idx=pad_idx,
        )
