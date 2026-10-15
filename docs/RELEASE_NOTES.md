# Curated release notes

Prepared on 2026-10-14 from the three selected entry points and their local dependencies. Original source hashes are listed in `SOURCE_MANIFEST.json`.

## Retained

- Segmentation and classification network module structure, parameter names, and forward computations, including modules defined but unused by the original forward pass.
- EfficientNet-B3 encoder and default ImageNet initialization for training.
- Segmentation cross entropy, Adam settings, paired geometric augmentation, and six-view logit-sum TTA.
- Classification affine/flip augmentation, AdamW settings, and stage schedule, including stage-boundary optimizer resets.

## Publication changes

- Replaced machine-specific paths and forced GPU indices with explicit command-line arguments. Moved execution behind `main()` guards. Removed unused wildcard/model imports and unrelated datasets/losses/model families.
- Added optional offline encoder initialization; evaluation avoids redundant ImageNet downloads. Optional profiling imports are confined to network-file profiling examples.
- Fixed segmentation validation averaging from `sum / batch_idx + 1` to a sample-weighted mean. Original IoU denominator was parenthesized incorrectly and could divide by zero on one validation batch.
- Replaced CUDA-only metric allocations and inconsistent NumPy scalar `.item()` assumptions with device-independent metrics. Validation/test include every sample, including a final incomplete batch.
- Enforced image/mask pairing by filename stem; reject unmatched/duplicate pairs and unsupported binary masks. Expand grayscale images to RGB, remove alpha, read non-image files safely by filtering image extensions, and resize masks with nearest-neighbor interpolation. This deliberately changes the original bilinear-mask boundary handling.
- Preserved `[0, 1]` RGB image scaling and the absence of ImageNet mean/std normalization. Removed unused classification coarse-mask reads. Classification list parsing supports paths with spaces and rejects invalid labels.
- Split classification into explicit train/test modes. Evaluation no longer builds the training and validation loaders or creates training output directories. CPU mode no longer forces a CUDA loss object.
- Added correct classification validation loss. Removed `labels.squeeze()` so batch-size-one cross entropy remains valid. CUDA AMP is conditional on the selected device.
- Replaced classification AUC computed from hard predictions with probability-based one-vs-rest ROC AUC for each class. Removed fabricated standard deviations. Export probabilities and all per-class metrics.
- Added strict checkpoint loading with `weights_only=True` and `map_location`; legacy raw classification state dictionaries and segmentation `net` dictionaries remain accepted.
- Export per-image/per-class segmentation metrics and both-class summaries; undefined ratios are explicit `nan`. Dice/IoU/precision/recall conventions are documented rather than conflating foreground and macro scores. New CSV names and definitions differ from the historical files.
- Added README, dependency pins, citation metadata, third-party notices, and output/credential exclusions. No license grant has been invented.

These changes are intended to make a focused code release inspectable and executable. The release is not a byte-for-byte archival copy or a claim that corrected metrics reproduce historical paper numbers. Original files remain unchanged on the source server.
