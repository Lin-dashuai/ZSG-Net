"""Train or evaluate the released SR-input binary ultrasound classifier."""
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from my_datasets_sr import MyDataSet_cls, MyValDataSet_cls
from N_cls4 import N_Unet
from zsg_utils import (binary_class_metrics, load_weights, resolve_device,
                       save_config, seed_everything, write_csv)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['train', 'test'], default='test')
    for split in ('train', 'val', 'test'):
        parser.add_argument(f'--{split}-root')
        parser.add_argument(f'--{split}-list')
    parser.add_argument('--checkpoint', default='checkpoints/classification/best.pth')
    parser.add_argument('--result-path', default='results/classification')
    parser.add_argument('--tb-path', default='runs/classification')
    parser.add_argument('--resize', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--test-batch-size', type=int, default=1)
    parser.add_argument('--stage-epochs', nargs='+', type=int, default=[50] * 7)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-decay', type=float, default=2)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=6666)
    parser.add_argument('--encoder-weights', choices=['imagenet', 'none'], default='imagenet')
    parser.add_argument('--no-amp', action='store_true')
    args = parser.parse_args()
    splits = ('train', 'val') if args.mode == 'train' else ('test',)
    for split in splits:
        if not getattr(args, split + '_root') or not getattr(args, split + '_list'):
            parser.error(f'--{split}-root and --{split}-list are required in {args.mode} mode.')
    if args.resize <= 0 or args.resize % 32 or min(args.batch_size, args.test_batch_size, *args.stage_epochs) < 1:
        parser.error('resize must be a positive multiple of 32; batch sizes and stage epochs must be positive.')
    if args.workers < 0 or args.lr <= 0 or args.lr_decay <= 0:
        parser.error('workers >= 0; learning rate and decay > 0.')
    return args


def make_loader(args, split, device):
    dataset_type = MyDataSet_cls if split == 'train' else MyValDataSet_cls
    dataset = dataset_type(getattr(args, split + '_root'), getattr(args, split + '_list'),
                           (args.resize, args.resize))
    return DataLoader(dataset, batch_size=args.test_batch_size if split == 'test' else args.batch_size,
                       shuffle=split == 'train', num_workers=args.workers,
                       pin_memory=device.type == 'cuda', drop_last=False)


def train(args, model, device):
    loaders = [make_loader(args, split, device) for split in ('train', 'val')]
    criterion = torch.nn.CrossEntropyLoss()
    amp = device.type == 'cuda' and not args.no_amp
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    lr = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-1)
    boundaries = set(np.cumsum(args.stage_epochs)[:-1].tolist())
    best_accuracy, history = -1.0, []
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_config(checkpoint.parent / 'config.json', args)
    with SummaryWriter(args.tb_path) as writer:
        for epoch in range(sum(args.stage_epochs)):
            row = dict(epoch=epoch)
            for split, loader in zip(('train', 'val'), loaders):
                model.train(split == 'train')
                total, correct, loss_sum = 0, 0, 0.0
                with torch.set_grad_enabled(split == 'train'):
                    for images, labels, _ in tqdm(loader, desc=split):
                        images, labels = images.to(device), labels.to(device)
                        if split == 'train':
                            optimizer.zero_grad()
                        with torch.autocast(device_type=device.type, enabled=amp):
                            logits = model(images)
                            loss = criterion(logits, labels)
                        if split == 'train':
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        total += labels.numel()
                        correct += int((logits.argmax(1) == labels).sum().item())
                        loss_sum += loss.item() * labels.numel()
                row[split + '_loss'] = loss_sum / total
                row[split + '_accuracy'] = correct / total
            if row['val_accuracy'] >= best_accuracy:
                best_accuracy = row['val_accuracy']
                torch.save(dict(net=model.state_dict(), epoch=epoch, accuracy=best_accuracy), checkpoint)
            history.append(row)
            write_csv(checkpoint.parent / 'history.csv', history)
            for name, value in row.items():
                if name != 'epoch':
                    writer.add_scalar(name, value, epoch)
            print(row)
            if epoch + 1 in boundaries:
                lr /= args.lr_decay
                # Original stage schedule resets AdamW state at each stage boundary.
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-1)


def evaluate(args, model, device):
    load_weights(model, args.checkpoint, device)
    model.eval()
    loader = make_loader(args, 'test', device)
    result = Path(args.result_path)
    result.mkdir(parents=True, exist_ok=True)
    save_config(result / 'config.json', args)
    rows, all_labels, all_probabilities = [], [], []
    with torch.inference_mode():
        for images, labels, names in tqdm(loader, desc='test'):
            probabilities = model(images.to(device)).softmax(1).cpu().numpy()
            for name, label, probability in zip(names, labels.tolist(), probabilities):
                rows.append(dict(image=name, target=label, prediction=int(probability.argmax()),
                                 probability_benign=float(probability[0]), probability_malignant=float(probability[1])))
            all_labels.extend(labels.tolist())
            all_probabilities.extend(probabilities)
    probabilities = np.asarray(all_probabilities)
    labels = np.asarray(all_labels)
    predictions = probabilities.argmax(1)
    metrics = binary_class_metrics(labels, predictions)
    for row, name in zip(metrics, ('Benign', 'Malignant')):
        class_id = row['class_id']
        target = (labels == class_id).astype(int)
        row['class_name'] = name
        row['auc'] = float(roc_auc_score(target, probabilities[:, class_id])) if len(np.unique(target)) == 2 else float('nan')
        row['average_precision'] = float(average_precision_score(target, probabilities[:, class_id])) if target.any() else float('nan')
    write_csv(result / 'predictions.csv', rows)
    write_csv(result / 'metrics_per_class.csv', metrics)
    confusion = np.bincount(2 * labels + predictions, minlength=4).reshape(2, 2)
    write_csv(result / 'confusion_matrix.csv', [dict(actual_class=i, predicted_0=int(row[0]),
                                                    predicted_1=int(row[1])) for i, row in enumerate(confusion)])
    print(metrics)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    weights = 'imagenet' if args.mode == 'train' and args.encoder_weights == 'imagenet' else None
    model = N_Unet(3, 2, encoder_weights=weights).to(device)
    if args.mode == 'train':
        train(args, model, device)
    else:
        evaluate(args, model, device)


if __name__ == '__main__':
    main()
