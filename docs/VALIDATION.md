# Validation

Checked on 2026-10-14 using an isolated Linux CPU workspace.

## Environment

- Python 3.10.18.
- PyTorch 2.6.0+cu126 and torchvision 0.21.0+cu126; execution used CPU only, with two CPU compute threads.
- segmentation-models-pytorch 0.3.3, efficientnet-pytorch 0.7.1, pretrainedmodels 0.7.4, and timm 0.9.2 installed into an isolated dependency directory.
- Other dependencies match the release baseline in `requirements.txt`.

The existing research environment and original server source files were not modified.

## Passed checks

1. Parse all 15 Python files and scan the release files for credential patterns and internal server addresses/paths.
2. Compare every network class against the original using Python ASTs. All class bodies match after normalizing the added encoder initialization option; forward computations and parameter definitions are retained.
3. Run `--help` for all three entry points.
4. Run one segmentation training/validation epoch on four/two synthetic 64x64 inputs, on CPU with random encoder initialization; save and reload the best checkpoint.
5. Run segmentation evaluation with six-view TTA and batch size two on two synthetic images. Confirm two PNG predictions, four per-image/per-class rows, fourteen per-class summary rows, and a confusion matrix.
6. Run one classification training/validation epoch with the same synthetic partition sizes. Save and reload the checkpoint, evaluate both classes, and confirm prediction probabilities, finite probability-based AUC, and both-class metric rows.
7. Load a legacy raw classification state dictionary strictly.
8. Check metric calculations against a hand-computed confusion example, absent-class `nan` behavior, grayscale/RGBA conversion to RGB, binary mask values, and inverse TTA alignment with a two-image batch.

All seven CLI invocations exited with status 0. The smoke runner, excluding dependency installation, took 31.46 seconds. Generated smoke data, checkpoints, predictions, and logs are excluded from the repository.

## Limits

The audit is limited to synthetic inputs and random initialization, with no patient data or paper checkpoints. It does not measure task performance or reproduce paper results.

CUDA/AMP execution, the optional ImageNet weight download, full-resolution training, formal dataset evaluation, and recovery from historical optimizer checkpoints were not exercised. Source-matching network classes do not imply that preprocessing and corrected metric outputs reproduce the original experiment numbers.
