"""Runtime helpers shared by the three released entry points."""
import csv
import json
from pathlib import Path
import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; pass --device cpu.')
    return device


def load_weights(model, path, device):
    # Load only trusted state dictionaries. Full serialized model objects are unsupported.
    state = torch.load(path, map_location=device, weights_only=True)
    if 'net' in state:
        weights = state['net']
    else:
        weights = state
    model.load_state_dict(weights, strict=True)
    return state


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_config(path, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vars(args), indent=2) + '\n', encoding='utf-8')


def binary_class_metrics(target, prediction):
    """One-vs-rest metrics, including both background/benign and foreground/malignant.

    Undefined ratios are NaN. Accuracy is one-vs-rest accuracy, equal for both
    classes in a binary problem. Dice and F1 are algebraically equivalent.
    """
    target, prediction = np.asarray(target).reshape(-1), np.asarray(prediction).reshape(-1)
    if target.shape != prediction.shape or not target.size:
        raise ValueError('Targets and predictions must have the same nonempty shape.')
    if not np.isin(target, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
        raise ValueError('This release supports binary labels 0 and 1.')

    def ratio(a, b):
        return float(a / b) if b else float('nan')

    rows = []
    for class_id in (0, 1):
        actual, pred = target == class_id, prediction == class_id
        tp = int(np.count_nonzero(actual & pred))
        fp = int(np.count_nonzero(~actual & pred))
        fn = int(np.count_nonzero(actual & ~pred))
        tn = int(np.count_nonzero(~actual & ~pred))
        rows.append(dict(class_id=class_id, tp=tp, fp=fp, fn=fn, tn=tn,
                         accuracy=ratio(tp + tn, target.size),
                         dice=ratio(2 * tp, 2 * tp + fp + fn),
                         iou=ratio(tp, tp + fp + fn), precision=ratio(tp, tp + fp),
                         recall=ratio(tp, tp + fn), specificity=ratio(tn, tn + fp),
                         f1=ratio(2 * tp, 2 * tp + fp + fn)))
    return rows
