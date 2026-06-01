from __future__ import annotations

import torch
from sklearn.metrics import classification_report, confusion_matrix

from torch.utils.data import DataLoader


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        preds = logits.argmax(1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

    return all_preds, all_labels


def evaluate(
    model: torch.nn.Module,
    data: dict,
    device: torch.device,
) -> tuple[list[int], list[int], str, ]:
    label_names = data.get("label_names")
    preds, labels = collect_predictions(model, data["test_loader"], device)

    report = classification_report(
        labels, preds, target_names=label_names, zero_division=0
    )
    cm = confusion_matrix(labels, preds)

    return preds, labels, report, cm
