from __future__ import annotations

import json
import time
from typing import cast

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, LRScheduler
from torch.utils.data import DataLoader

from .config import HyperParams


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    hp: HyperParams,
    train_loader: DataLoader,
) -> LRScheduler:
    total_steps = len(train_loader) * hp.epochs
    
    return OneCycleLR(
        optimizer,
        max_lr=hp.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        cycle_momentum=True,
        div_factor=25.0,
        final_div_factor=1e4
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    scheduler: LRScheduler,
    device: torch.device,
    hp: HyperParams,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)

        if hasattr(model, "path_reg_loss"):
            if hp.lambda_path > 0:
                reg = model.path_reg_loss()
                loss = loss + hp.lambda_path * reg

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    return running_loss / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        loss = criterion(logits, labels)

        running_loss += loss.item() * inputs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += inputs.size(0)

    return running_loss / total, correct / total


def train(
    model: nn.Module,
    data: dict,
    device: torch.device,
    hp: HyperParams,
    model_path: str,
    history_path: str
) -> None:
    criterion = nn.CrossEntropyLoss(label_smoothing=hp.label_smoothing)
    optimizer = AdamW(
        model.parameters(),
        lr=hp.lr,
        betas=(hp.beta1, hp.beta2),
        weight_decay=hp.weight_decay,
    )
    scheduler = build_scheduler(optimizer, hp, cast(DataLoader, data["train_loader"]))

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state: dict | None = None

    print("Starting training")

    for epoch in range(1, hp.epochs + 1):
        t0 = time.perf_counter()

        train_loss = train_epoch(
            model, data["train_loader"], criterion, optimizer, scheduler, device, hp
        )
        val_loss, val_acc = evaluate(
            model, data["test_loader"], criterion, device
        )
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(f"Epoch {epoch}/{hp.epochs} | train_loss={train_loss:4f} | val_loss={val_loss:4f} | val_acc={val_acc:4f} | lr={current_lr:4e} | {elapsed:1f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = model.state_dict()
            torch.save(best_state, model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= hp.patience:
                print("Early stopping at epoch", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
