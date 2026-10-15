# ZSG-Net

Ultrasound super-resolution, segmentation, and classification code associated with:

**ZSG-Net: A Zero-Shot Super-Resolution Guided Network for Ultrasound Image Segmentation and Classification**
Xingtao Lin, Xiahai Zhuang, Lin Pan, Mingjing Yang, Liqin Huang, Shun Chen, and Lei Li.
*IEEE Journal of Biomedical and Health Informatics*, 29(11), 8320–8333, 2025.
[Paper DOI](https://doi.org/10.1109/JBHI.2025.3584505)

本仓库发布超分辨率入口 `ZSG-SR.py`、三个指定的分割/分类入口及必要依赖；不包含数据集和训练权重。所有 SR 代码与主训练文件同级。

## Release scope

This is a curated, portable release of the selected super-resolution (SR), segmentation, and classification experiment entry points and their necessary dependencies. `ZSG-SR.py` performs per-image cycle training with random image degradation (RID) and saves reconstructed images. Downstream scripts consume the resulting images or other explicitly supplied inputs. Data preparation, historical dataset partitions, paper checkpoints, and reproduced benchmark results are not distributed.

| Entry point | Purpose | Network |
| --- | --- | --- |
| `ZSG-SR.py` | Per-image cycle training and fourfold SR reconstruction | `ZSG.ZSG`, `N_Shape04.N_Unet`, and `sr_models` |
| `TNSCUI_195-512_train.py` | Binary segmentation training and validation | `N_Shape04.N_Unet` |
| `TNSCUI_780-512_test.py` | Binary segmentation evaluation; six-view test-time augmentation by default | `N_Shape04.N_Unet` |
| `TNSCUI_N04sr_CLS.py` | Binary classification training or evaluation | `N_cls4.N_Unet` |

The original entry-point filenames are retained for traceability. The segmentation training script's historical defaults pointed to a TNUI directory despite its TNSCUI filename; dataset paths are now explicit arguments. No dataset identity, split, or sample count is inferred from a filename.

The segmentation and classification networks retain their original module structure, parameter names, and forward computations. Their EfficientNet-B3 encoders are provided by `segmentation_models_pytorch`. An `encoder_weights` option permits offline initialization without changing the architecture. SR network definitions and the active four-term loss are retained from the selected SR experiment. Runtime, data handling, and evaluation were cleaned up; see [release notes](docs/RELEASE_NOTES.md) for behavioral differences.

## Installation

Use a dedicated Python 3.10 environment. The release is checked against PyTorch 2.6.0 and torchvision 0.21.0; it does not claim these were the historical paper-training versions.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Choose a compatible CPU/CUDA PyTorch build for your machine first if needed.
python -m pip install -r requirements.txt
```

For Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. The optional network-file profiling examples require `thop`; this package is not needed by the three entry points.

Training defaults to ImageNet encoder initialization, which may download pretrained encoder weights. Pass `--encoder-weights none` for offline random initialization. Evaluation constructs the encoder without downloading weights and loads the supplied complete network checkpoint.

The SR entry point uses `--pretrained imagenet` by default for both encoders and VGG19 features. Its `--pretrained none` option is for offline wiring checks; randomly initialized perceptual features do not establish meaningful reconstruction quality.

The scripts honor externally assigned visible GPUs and never set `CUDA_VISIBLE_DEVICES`. Use `--device cpu` for CPU execution or `--device cuda` for the visible CUDA device. `--device auto` is the default.

## Dataset format

Use existing dataset locations; no project-specific dataset copy is needed. All inputs are read as three-channel RGB and scaled to `[0, 1]`, without ImageNet mean/std normalization, matching the original input convention. Inputs are resized to a square (default `512`; a positive multiple of `32`).

For segmentation, provide separate image/mask directories with matching, unique filename stems. Example:

```text
/path/to/dataset/
  train/image/case001.png
  train/mask/case001.png
  val/image/case002.png
  val/mask/case002.png
  test/image/case003.png
  test/mask/case003.png
```

Masks must be single-channel binary images: `0/255` or `0/1`. Class `0` is background; class `1` is foreground. Masks use nearest-neighbor resizing. The release checks pairing by stem rather than silently pairing independently sorted lists. Training uses paired random horizontal/vertical flips and 90-degree rotations.

Classification lists are UTF-8 text files containing one image path relative to the corresponding image root and its integer class label per line:

```text
case001.png 0
subfolder/case002.png 1
```

Class `0` is benign; class `1` is malignant. Classification training keeps the original affine/flip augmentation; validation and test use deterministic bilinear resizing. Use your own verified training/validation/test partitions; no data or split lists are distributed here.

## Super-resolution reconstruction

```bash
python ZSG-SR.py \
  --input-path /path/to/input_pngs \
  --result-path results/super_resolution --device cuda
```

The input can be one PNG or a directory of PNGs. Images ending in `_gt.png` are reserved as optional aligned ground truths and are excluded from the input loop. Each input is adapted independently with a freshly initialized model; inputs are processed sequentially on the requested visible device. There is no GPU polling, automatic GPU selection, or subprocess queue.

The released experiment first resizes each input to **192x192 RGB**, then reconstructs a **768x768** image at a fixed fourfold scale. The learned degradation network has a fixed 192-element noise input and a 192x192 noise map; arbitrary input geometry and gradual/multiple scales are not advertised as supported. This is the configuration found in the selected source, irrespective of the original image resolution.

Defaults are `3000` iterations per image, Adam learning rate `0.001`, eight-view geometric self-ensemble, and `10` back-projection iterations per view and after fusion. The active objective is:

```text
loss = cycle_loss + l1_loss + feature_loss + 0.15 * gan_loss
```

VGG19 uses the original first 28 feature layers, frozen parameters, 224x224 feature inputs, and the original `(x + 1) / 2` feature preprocessing. Unused alternative loss expressions and model families are excluded. `--max-iters`, `--no-ensemble`, `--back-projection-iters`, `--run-test-every`, and `--display-every` expose runtime controls. A numeric `<stem>_0.mat` file, when present, must contain a two-dimensional `Kernel` and is used by the original kernel/back-projection path; RID training remains independently sampled.

Outputs in `--result-path` are:

- `<stem>_zsg_X4.00X4.00.png`: reconstructed RGB prediction, stored losslessly as an 8-bit PNG.
- `<stem>_losses.csv`: total, cycle, L1, feature, GAN, and both discriminator losses at every completed iteration, plus the effective learning rate.
- `metrics.csv`: image/prediction names, aligned-GT availability, RGB PSNR and SSIM, timings, and completed iterations.
- `config.json`: effective configuration. Optional `--save-checkpoint` saves each image-specific model state dictionary; these can be large and are excluded from Git.

For quantitative evaluation, place an aligned **768x768** `<stem>_gt.png` alongside the corresponding input. Ground truth is never silently resized for scoring. Metrics use the saved 8-bit reconstruction: PSNR has data range 1 after conversion to `[0, 1]`; SSIM is the mean of three channel scores with an 11x11, sigma-1.5 Gaussian window and its five-pixel valid border. Without ground truth, PSNR/SSIM are explicitly `nan`. SR has no semantic classes; class-specific metrics belong to the downstream tasks.

SR modules are all at repository root: `ZSG.py`, `sr_config.py`, `sr_models.py`, `sr_vgg.py`, `sr_degradation.py`, `sr_image.py`, `sr_resize.py`, and `sr_utils.py`. The original per-input invocation is integrated into `ZSG-SR.py`. Active module names, class names, imports, and output names use `ZSG`; source filenames in the provenance manifest retain their historical spelling for traceability.

## Segmentation

Train:

```bash
python TNSCUI_195-512_train.py \
  --imgs_train_path /path/to/train/image \
  --labels_train_path /path/to/train/mask \
  --imgs_val_path /path/to/val/image \
  --labels_val_path /path/to/val/mask \
  --checkpoint checkpoints/segmentation/best.pth \
  --tb_path runs/segmentation --device cuda
```

Defaults: resize `512`, batch size `8`, Adam learning rate `1e-4`, betas `(0.9, 0.99)`, `300` epochs. The best checkpoint uses corrected per-image macro validation IoU, expressed as a percentage. `--resume` reloads that checkpoint; new checkpoints also store Adam state. `--checkpoint` now accepts a file path rather than a directory.

Evaluate:

```bash
python TNSCUI_780-512_test.py \
  --imgs_test_path /path/to/test/image \
  --labels_test_path /path/to/test/mask \
  --checkpoint checkpoints/segmentation/best.pth \
  --result_path results/segmentation --device cuda
```

The default TTA sums aligned logits for the original view, rotations by 90/180/270 degrees, horizontal flip, and vertical flip. This matches the original six-view computation and supports batches larger than one. Use `--no-tta` to disable it.

Outputs:

- `predictions/*.png`: lossless `0/255` masks at the evaluation resolution.
- `per_image_per_class.csv`: each image, both classes, confusion counts, accuracy, Dice, IoU, precision, recall/sensitivity, specificity, and F1.
- `summary_per_class.csv`: per-class mean, population standard deviation, valid-image count, and total-image count for every metric.
- `confusion_matrix.csv`: dataset-level pixel confusion counts, with actual classes in rows and predicted classes in columns.
- `config.json`: the invocation configuration.

## Classification

Train:

```bash
python TNSCUI_N04sr_CLS.py --mode train \
  --train-root /path/to/train/image --train-list /path/to/Train.txt \
  --val-root /path/to/val/image --val-list /path/to/Val.txt \
  --checkpoint checkpoints/classification/best.pth --device cuda
```

Defaults: resize `512`, batch size `16`, AdamW learning rate `1e-4`, weight decay `0.1`, seven stages of `50` epochs. The learning rate is divided by two and AdamW state is reset at each stage boundary, preserving the original schedule. The best validation-accuracy checkpoint is retained. CUDA AMP is enabled on CUDA by default; use `--no-amp` to disable it. Test mode is the default; it does not load training/validation data or create a training writer.

Evaluate:

```bash
python TNSCUI_N04sr_CLS.py --mode test \
  --test-root /path/to/test/image --test-list /path/to/Test.txt \
  --checkpoint checkpoints/classification/best.pth \
  --result-path results/classification --device cuda
```

Outputs:

- `predictions.csv`: image name, true label, predicted label, and probabilities for both classes.
- `metrics_per_class.csv`: confusion counts, accuracy, Dice/F1, IoU, precision, recall/sensitivity, specificity, probability-based ROC AUC, and average precision for both classes.
- `confusion_matrix.csv` and `config.json`.

Only trusted state-dictionary checkpoints are supported. Segmentation and new classification checkpoints use `{'net': state_dict, ...}`; legacy raw classification state dictionaries are also supported. Loading is strict so missing or mismatched weights cannot silently pass.

## Metric conventions and validation limits

All reported metrics are fractions in `[0, 1]`, except training segmentation IoU fields explicitly named `*_percent`. Accuracy is one-vs-rest accuracy; in a binary task it is equal for the two classes. An undefined ratio is written as `nan`, never replaced by a fabricated zero. Segmentation summaries exclude undefined values and report their valid counts. ROC AUC is `nan` if the test set lacks either class. A single run does not estimate an experimental standard deviation; classification does not add placeholder `±0.000` values.

The publication audits use synthetic data and randomly initialized models to check training, checkpoint loading, inference, TTA/ensemble alignment, reconstruction predictions, and metric exports. They do **not** validate paper scores or establish parity with the original experiments. See [validation](docs/VALIDATION.md).

## Citation and rights

Use GitHub's citation button or `CITATION.cff` for the paper reference.

```bibtex
@article{lin2025zsgnet,
  author = {Lin, Xingtao and Zhuang, Xiahai and Pan, Lin and Yang, Mingjing and Huang, Liqin and Chen, Shun and Li, Lei},
  title = {{ZSG-Net}: A Zero-Shot Super-Resolution Guided Network for Ultrasound Image Segmentation and Classification},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year = {2025}, volume = {29}, number = {11}, pages = {8320--8333},
  doi = {10.1109/JBHI.2025.3584505}
}
```

No project-wide license has been selected for this release. Public availability does not itself grant an open-source license. External libraries and adapted upstream source retain their own licenses; the BSRGAN license is included in `LICENSE-BSRGAN.txt`. See [third-party notices](THIRD_PARTY_NOTICES.md) for SR/BSRGAN attribution. Datasets, trained checkpoints, the paper PDF, and experiment outputs are not included.
