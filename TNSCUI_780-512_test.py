"""Evaluate binary segmentation with the original six-view logit-sum TTA."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.create_dataset import CreateDataset
from N_Shape04 import N_Unet
from zsg_utils import (binary_class_metrics, load_weights, resolve_device,
                       save_config, seed_everything, write_csv)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imgs_test_path', '-it', required=True)
    parser.add_argument('--labels_test_path', '-lt', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--result_path', default='results/segmentation')
    parser.add_argument('--resize', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=6666)
    parser.add_argument('--no-tta', action='store_true')
    args = parser.parse_args()
    if args.resize <= 0 or args.resize % 32 or args.batch_size < 1 or args.workers < 0:
        parser.error('resize must be a positive multiple of 32; batch_size > 0; workers >= 0.')
    return args


def predict_logits(model, images, tta=True):
    if not tta:
        return model(images)
    views = [images] + [torch.rot90(images, k, [2, 3]) for k in (1, 2, 3)]
    views += [torch.flip(images, [-1]), torch.flip(images, [-2])]
    logits = model(torch.cat(views)).chunk(6)
    return (logits[0] + torch.rot90(logits[1], 3, [2, 3]) + torch.rot90(logits[2], 2, [2, 3])
            + torch.rot90(logits[3], 1, [2, 3]) + torch.flip(logits[4], [-1]) + torch.flip(logits[5], [-2]))


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    result = Path(args.result_path)
    (result / 'predictions').mkdir(parents=True, exist_ok=True)
    save_config(result / 'config.json', args)
    dataset = CreateDataset(args.imgs_test_path, args.labels_test_path, args.resize, 'val')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == 'cuda')
    model = N_Unet(3, 2, encoder_weights=None).to(device)
    load_weights(model, args.checkpoint, device)
    model.eval()
    per_image, total_confusion = [], np.zeros((2, 2), dtype=np.int64)
    with torch.inference_mode():
        for images, targets, names in tqdm(loader, desc='test'):
            predictions = predict_logits(model, images.to(device), not args.no_tta).argmax(1).cpu().numpy()
            for prediction, target, name in zip(predictions, targets.numpy(), names):
                for row in binary_class_metrics(target, prediction):
                    per_image.append(dict(image=Path(name).name, **row))
                total_confusion += np.bincount((2 * target + prediction).reshape(-1), minlength=4).reshape(2, 2)
                destination = result / 'predictions' / (Path(name).stem + '.png')
                if not cv2.imwrite(str(destination), (prediction * 255).astype(np.uint8)):
                    raise OSError(f'Failed to save prediction: {destination}')
    write_csv(result / 'per_image_per_class.csv', per_image)
    metric_names = ['accuracy', 'dice', 'iou', 'precision', 'recall', 'specificity', 'f1']
    summary = []
    for class_id in (0, 1):
        for metric in metric_names:
            values = np.array([row[metric] for row in per_image if row['class_id'] == class_id])
            valid = values[np.isfinite(values)]
            summary.append(dict(class_id=class_id, metric=metric,
                                mean=float(valid.mean()) if valid.size else float('nan'),
                                std=float(valid.std()) if valid.size else float('nan'),
                                valid_images=int(valid.size), total_images=int(values.size)))
    write_csv(result / 'summary_per_class.csv', summary)
    write_csv(result / 'confusion_matrix.csv', [dict(actual_class=i, predicted_0=int(row[0]),
                                                    predicted_1=int(row[1])) for i, row in enumerate(total_confusion)])
    print(f'Saved predictions and metrics for {len(dataset)} images to {result}')


if __name__ == '__main__':
    main()
