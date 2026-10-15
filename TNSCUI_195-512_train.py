"""Train the released binary segmentation network on supplied ultrasound images."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from data.create_dataset import CreateDataset
from models.loss.Loss_all import CrossEntropyLoss2D
from models.metrics.Miou import calculate_miou
from N_Shape04 import N_Unet
from zsg_utils import load_weights, resolve_device, save_config, seed_everything, write_csv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imgs_train_path', '-it', required=True)
    parser.add_argument('--labels_train_path', '-lt', required=True)
    parser.add_argument('--imgs_val_path', '-iv', required=True)
    parser.add_argument('--labels_val_path', '-lv', required=True)
    parser.add_argument('--resize', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--start_epoch', '-s', type=int, default=0)
    parser.add_argument('--end_epoch', '-e', type=int, default=300)
    parser.add_argument('--times', '-t', type=int, default=1, help='Validate every N epochs.')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N within visible devices.')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=6666)
    parser.add_argument('--tb_path', default='runs/segmentation')
    parser.add_argument('--checkpoint', default='checkpoints/segmentation/best.pth', help='Checkpoint file.')
    parser.add_argument('--resume', '-r', action='store_true')
    parser.add_argument('--encoder-weights', choices=['imagenet', 'none'], default='imagenet')
    args = parser.parse_args()
    if args.resize <= 0 or args.resize % 32 or min(args.batch_size, args.times) < 1 or args.workers < 0:
        parser.error('resize must be a positive multiple of 32; batch_size/times must be positive; workers >= 0.')
    if args.start_epoch < 0 or args.end_epoch <= args.start_epoch:
        parser.error('end_epoch must exceed nonnegative start_epoch.')
    return args


def run_epoch(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    loss_sum, miou_sum, count = 0.0, 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, targets, _ in tqdm(loader, desc='train' if optimizer else 'val'):
            images, targets = images.to(device), targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            size = images.shape[0]
            loss_sum += float(loss.item()) * size
            miou_sum += float(calculate_miou(logits.detach().argmax(1), targets).item()) * size
            count += size
    return loss_sum / count, 100.0 * miou_sum / count


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_config(checkpoint.parent / 'config.json', args)
    loaders = []
    for phase in ('train', 'val'):
        dataset = CreateDataset(getattr(args, f'imgs_{phase}_path'), getattr(args, f'labels_{phase}_path'),
                                args.resize, phase, aug=phase == 'train')
        loaders.append(DataLoader(dataset, batch_size=args.batch_size, shuffle=phase == 'train',
                                  num_workers=args.workers, pin_memory=device.type == 'cuda'))
    weights = None if args.resume or args.encoder_weights == 'none' else 'imagenet'
    model = N_Unet(3, 2, encoder_weights=weights).to(device)
    criterion = CrossEntropyLoss2D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    best_miou = -np.inf
    if args.resume:
        state = load_weights(model, checkpoint, device)
        best_miou = state.get('miou', -np.inf)
        args.start_epoch = state.get('epoch', args.start_epoch - 1) + 1
        if 'optimizer' in state:
            optimizer.load_state_dict(state['optimizer'])
    history = []
    with SummaryWriter(args.tb_path) as writer:
        for epoch in range(args.start_epoch, args.end_epoch):
            train_loss, train_miou = run_epoch(model, loaders[0], criterion, device, optimizer)
            row = dict(epoch=epoch, train_loss=train_loss, train_miou_percent=train_miou,
                       val_loss=float('nan'), val_miou_percent=float('nan'))
            if epoch % args.times == 0:
                val_loss, val_miou = run_epoch(model, loaders[1], criterion, device)
                row.update(val_loss=val_loss, val_miou_percent=val_miou)
                if val_miou > best_miou:
                    best_miou = val_miou
                    torch.save(dict(net=model.state_dict(), optimizer=optimizer.state_dict(),
                                    miou=best_miou, epoch=epoch), checkpoint)
            for name, value in row.items():
                if name != 'epoch':
                    writer.add_scalar(name, value, epoch)
            history.append(row)
            write_csv(checkpoint.parent / 'history.csv', history)
            print(row)


if __name__ == '__main__':
    main()
