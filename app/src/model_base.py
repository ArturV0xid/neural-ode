from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import HyperParams


class MiniBERT(nn.Module):
    def __init__(
        self,
        hp: HyperParams,
        num_classes: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.hp = hp
        self.num_classes = num_classes if num_classes is not None else hp.num_classes
        self.d_model = hp.d_model

        self.tok_emb = nn.Embedding(hp.vocab_size, hp.d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(hp.max_seq_len, hp.d_model)
        self.emb_dropout = nn.Dropout(hp.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hp.d_model,
            nhead=hp.nheads,
            dim_feedforward=hp.dff,
            dropout=hp.dropout,
            activation=hp.activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=hp.num_layers,
        )

        self.norm = nn.LayerNorm(hp.d_model)
        self.classifier = nn.Linear(hp.d_model, self.num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape
        device = input_ids.device

        pad_mask = input_ids == 0
        positions = torch.arange(seq_len, device=device)

        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        cls_repr = x[:, 0]
        cls_repr = self.norm(cls_repr)
        logits = self.classifier(cls_repr)
        return logits
