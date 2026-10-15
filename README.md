# ZSG-Net

Downstream ultrasound segmentation and classification code associated with:

**ZSG-Net: A Zero-Shot Super-Resolution Guided Network for Ultrasound Image Segmentation and Classification**
Xingtao Lin, Xiahai Zhuang, Lin Pan, Mingjing Yang, Liqin Huang, Shun Chen, and Lei Li.
*IEEE Journal of Biomedical and Health Informatics*, 29(11), 8320–8333, 2025.
[Paper DOI](https://doi.org/10.1109/JBHI.2025.3584505)

本仓库发布指定的三个分割/分类入口及其必要依赖；不包含超分辨率生成阶段、数据集和训练权重。

## Release scope

This is a curated, portable release of the three selected experiment entry points and their necessary local dependencies. It accepts images that have already been prepared, including super-resolved images supplied by the user. **The ZSCycle-GAN super-resolution training/inference stage and RID implementation are outside this release.** This repository alone does not reproduce the complete paper pipeline.

| Entry point | Purpose | Network |
| --- | --- | --- |
| `TNSCUI_195-512_train.py` | Binary segmentation training and validation | `N_Shape04.N_Unet` |
| `TNSCUI_780-512_test.py` | Binary segmentation evaluation; six-view test-time augmentation by default | `N_Shape04.N_Unet` |
| `TNSCUI_N04sr_CLS.py` | Binary classification training or evaluation | `N_cls4.N_Unet` |

The original entry-point filenames are retained for traceability. The segmentation training script's historical defaults pointed to a TNUI directory despite its TNSCUI filename; dataset paths are now explicit arguments. No dataset identity, split, or sample count is inferred from a filename.

Both networks retain their original module structure, parameter names, and forward computations. Their EfficientNet-B3 encoders are provided by `segmentation_models_pytorch`. A new `encoder_weights` option permits offline initialization without changing the architecture. Data handling, runtime configuration, and evaluation were cleaned up; see [release notes](docs/RELEASE_NOTES.md) for behavioral differences.

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

The publication audit uses synthetic data and randomly initialized models to check training, checkpoint loading, inference, TTA, predictions, and metric exports. It does **not** validate paper scores or establish parity with the original experiments. See [validation](docs/VALIDATION.md).

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

No project-wide license has been selected for this release. Public availability does not itself grant an open-source license. External libraries retain their own licenses; see [third-party notices](THIRD_PARTY_NOTICES.md). Datasets, trained checkpoints, the paper PDF, and experiment outputs are not included.
