from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HyperParams:
    vocab_size: int = 10_000
    d_model: int = 256
    nheads: int = 8
    num_layers: int = 4
    dff: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 128
    activation: str = "gelu"
    num_classes: int = 20

    integrator: str = "dopri5"
    rtol: float = 1e-4
    atol: float = 1e-4
    options: dict | None = None

    lr: float = 2e-4
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 1e-2
    label_smoothing: float = 0.1
    lambda_path: float = 1e-3

    batch_size: int = 64
    epochs: int = 20
    patience: int = 5
