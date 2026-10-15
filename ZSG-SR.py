"""Run per-image ZSG fourfold super-resolution cycle training and reconstruction."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from sr_config import Config
from zsg_utils import seed_everything, write_csv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-path', required=True, help='Input PNG or directory of PNG images. Paired ground truths use <stem>_gt.png.')
    parser.add_argument('--result-path', default='results/super_resolution')
    parser.add_argument('--device', default='auto', help='cpu, cuda, cuda:N, or auto within externally visible devices.')
    parser.add_argument('--max-iters', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=6666)
    parser.add_argument('--pretrained', choices=['imagenet', 'none'], default='imagenet',
                        help='none is intended for offline wiring checks; meaningful perceptual training requires pretrained VGG features.')
    parser.add_argument('--no-ensemble', action='store_true')
    parser.add_argument('--back-projection-iters', type=int, default=10)
    parser.add_argument('--run-test-every', type=int, default=60)
    parser.add_argument('--display-every', type=int, default=50)
    parser.add_argument('--save-checkpoint', action='store_true', help='Save each image-specific model state dictionary; files can be large.')
    args = parser.parse_args()
    if min(args.max_iters, args.run_test_every, args.display_every) < 1 or args.back_projection_iters < 0:
        parser.error('Iteration counts must be positive; back projection iterations must be nonnegative.')
    return args


def quality_metrics(output, ground_truth):
    row = dict(psnr=float('nan'), ssim=float('nan'))
    if ground_truth is None:
        return row
    if output.shape != ground_truth.shape:
        raise ValueError('SR and aligned ground truth shapes differ.')
    squared_error = np.mean((output.astype(np.float64) - ground_truth) ** 2)
    row['psnr'] = float('inf') if squared_error == 0 else float(-10 * np.log10(squared_error))
    window = cv2.getGaussianKernel(11, 1.5)
    window = window @ window.T
    values = []
    for channel in range(3):
        x, y = output[..., channel].astype(np.float64), ground_truth[..., channel].astype(np.float64)
        mu_x, mu_y = [cv2.filter2D(image, -1, window)[5:-5, 5:-5] for image in (x, y)]
        var_x = cv2.filter2D(x*x, -1, window)[5:-5, 5:-5] - mu_x**2
        var_y = cv2.filter2D(y*y, -1, window)[5:-5, 5:-5] - mu_y**2
        covariance = cv2.filter2D(x*y, -1, window)[5:-5, 5:-5] - mu_x*mu_y
        score = ((2*mu_x*mu_y + 0.01**2) * (2*covariance + 0.03**2)
                 / ((mu_x**2 + mu_y**2 + 0.01**2) * (var_x + var_y + 0.03**2)))
        values.append(score.mean())
    row['ssim'] = float(np.mean(values))
    return row


def main():
    args = parse_args()
    source, result = Path(args.input_path), Path(args.result_path)
    files = sorted(source.glob('*.png')) if source.is_dir() else [source]
    files = [path for path in files if path.is_file() and path.suffix.lower() == '.png' and not path.stem.endswith('_gt')]
    if not files:
        raise ValueError('No input PNG images found.')
    if len({path.stem for path in files}) != len(files):
        raise ValueError('Input filename stems must be unique.')
    result.mkdir(parents=True, exist_ok=True)
    weights = None if args.pretrained == 'none' else 'imagenet'
    conf = Config(input_path=str(source), result_path=str(result), device=args.device,
                  encoder_weights=weights, vgg_weights=weights, max_iters=args.max_iters,
                  output_flip=not args.no_ensemble, back_projection_iters=[args.back_projection_iters],
                  run_test_every=args.run_test_every, display_every=args.display_every,
                  save_checkpoint=args.save_checkpoint, seed=args.seed)
    (result/'config.json').write_text(json.dumps(asdict(conf), indent=2) + '\n', encoding='utf-8')
    from ZSG import ZSG
    rows = []
    for path in files:
        seed_everything(conf.seed)
        ground_truth = path.with_name(path.stem + '_gt.png')
        kernels = [path.with_name(path.stem + '_0.mat')]
        net = ZSG(str(path), conf, str(ground_truth) if ground_truth.is_file() else None,
                  [str(kernels[0])] if kernels[0].is_file() else None)
        prediction = net.run()
        prediction = np.round(np.clip(prediction, 0, 1) * 255).astype(np.float32) / 255.0
        rows.append(dict(image=path.name, prediction=Path(net.output_file).name,
                         has_ground_truth=net.gt is not None, **quality_metrics(prediction, net.gt),
                         training_seconds=net.timings['training_seconds'],
                         reconstruction_seconds=net.timings['reconstruction_seconds'],
                         completed_iterations=len(net.history)))
        write_csv(result/(path.stem + '_losses.csv'), net.history)
        write_csv(result/'metrics.csv', rows)
        del net
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f'Saved {len(rows)} reconstructed images and metric rows to {result}')


if __name__ == '__main__':
    main()
