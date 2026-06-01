from __future__ import annotations

import torch
import torch.nn as nn

from .config import HyperParams

from torchdiffeq import odeint_adjoint


class ODEFunc(nn.Module):
    def __init__(self, encoder: nn.TransformerEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.pad_mask: torch.Tensor | None = None

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        out = self.encoder(x, src_key_padding_mask=self.pad_mask)
        dx_dt = out - x
        return dx_dt


class MiniBERTNeuralODE(nn.Module):
    def __init__(self, hp: HyperParams) -> None:
        super().__init__()
        self.hp = hp

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
        encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=hp.num_layers,
        )

        self.ode_func = ODEFunc(encoder)
        self.integration_time = torch.tensor([0, 1]).float()
        self.method = hp.integrator
        self.rtol = hp.rtol
        self.atol = hp.atol
        self.options = hp.options

        self.norm = nn.LayerNorm(hp.d_model)
        self.classifier = nn.Linear(hp.d_model, hp.num_classes)

        self.last_trajectory: torch.Tensor | None = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape
        device = input_ids.device

        pad_mask = input_ids == 0

        positions = torch.arange(seq_len, device=device)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        self.ode_func.pad_mask = pad_mask
        t = self.integration_time.to(device)
        trajectory = odeint_adjoint(
            self.ode_func,
            x,
            t,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            options=self.options,
        )
        self.last_trajectory = trajectory
        x = trajectory[-1]

        cls_repr = x[:, 0]
        cls_repr = self.norm(cls_repr)
        logits = self.classifier(cls_repr)
        return logits

    def path_reg_loss(self) -> torch.Tensor:
        if self.last_trajectory is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        with torch.enable_grad():
            derivs = torch.stack([
                self.ode_func(self.integration_time[i], self.last_trajectory[i])
                for i in range(self.last_trajectory.shape[0])
            ])
        return (derivs ** 2).sum(dim=(2, 3)).mean()
