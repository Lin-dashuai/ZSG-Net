# Curated release notes

Initial downstream release prepared on 2026-10-14; SR extension prepared on 2026-10-15. Original source hashes and public-name mappings are listed in `SOURCE_MANIFEST.json`.

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

## Super-resolution extension, 2026-10-15

- Added `ZSG-SR.py` at the same level as the three downstream entry points. The original batch and per-input invocations are combined into this explicit, sequential CLI. The core module/class is `ZSG.ZSG`; active old module/class/output names are replaced.
- Added eight dependency modules at repository root. No SR subdirectory is introduced. Removed commented-out code and unused TensorFlow, GPUtil, profiling, plotting, model, and utility dependencies. Retained explanatory and upstream attribution docstrings.
- Removed comments from the shared `N_Shape04.py` generator while preserving its complete Python AST and network behavior.
- Reused the existing `N_Shape04.N_Unet(3, 3)` generator. The source SR variant's encoder used `in_channels=in_ch`; with the actual `in_ch=3` call this matches the existing encoder exactly. The degradation network, discriminator classes, module parameter definitions, and active loss expressions are preserved, apart from optional encoder initialization and discarded unused feature aliases.
- Preserved the source's 192x192 training image, 192-element degradation noise vector, fourfold reconstruction, 224x224 VGG features, loss weights `(1, 1, 1, 0.15)`, RID, replay buffers, median self-ensemble, and back projection. Unsupported arbitrary image geometry/multiple-scale configurations are rejected explicitly.
- Moved all tensor allocation and modules to the selected device. Removed hardcoded CUDA tensor construction, forced device visibility, asynchronous shell launches, and GPU availability polling. ImageNet weights remain the training default; `--pretrained none` is an explicit offline check option.
- Replaced removed `interp2d` with bilinear `RectBivariateSpline`, `scipy.finfo` with NumPy's equivalent, deprecated SciPy namespace calls with current calls, and removed NumPy integer aliases. Resize field-of-view indices now remain signed so negative reflection coordinates do not wrap as unsigned integers. Fixed the numeric-kernel loop to iterate channels and corrected the rectangular mod-crop height/width order.
- Evaluation switches the generator to eval/inference mode and restores its training state; historical inference kept training-mode batch normalization. Stored loss histories are detached floats, preventing retention of old autograd graphs. Learning-rate changes now update optimizer parameter groups; the historical code only changed a Python attribute. These intentional corrections can change results relative to historical runs.
- Added RGB input normalization, explicit aligned-ground-truth shape checks, lossless reconstruction PNGs, effective configuration, per-iteration component losses, timings, and PSNR/SSIM on saved predictions. No-ground-truth scores are `nan`. Optional checkpoints contain model state dictionaries.
- Added SciPy to requirements, source hashes, upstream SR/BSRGAN attribution, and the BSRGAN Apache-2.0 license. No blanket project-wide license grant is inferred.
