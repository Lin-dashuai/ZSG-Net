"""Device-independent per-image macro IoU, excluding absent classes."""
import torch


def calculate_miou(prediction, target, classNum=2):
    batch = []
    for pred, actual in zip(prediction, target):
        scores = []
        for class_id in range(classNum):
            p, t = pred == class_id, actual == class_id
            union = (p | t).sum()
            if union.item():
                scores.append((p & t).sum().float() / (union + 1e-6))
        batch.append(torch.stack(scores).mean())
    return torch.stack(batch).mean()
