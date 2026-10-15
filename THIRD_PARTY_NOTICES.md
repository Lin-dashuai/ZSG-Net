# Third-party components

Most dependencies are imported through the Python environment. The SR extension also includes selected adapted utility routines; their notices follow below. No pretrained weights are distributed.

- [PyTorch](https://github.com/pytorch/pytorch) and [torchvision](https://github.com/pytorch/vision): tensor operations, optimization, and image transformations.
- [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch): EfficientNet-B3 encoder factory.
- [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch): encoder implementation used by the pinned SMP release.
- [timm](https://github.com/huggingface/pytorch-image-models) and [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch): dependencies of the pinned SMP release.
- NumPy, SciPy, OpenCV, Pillow, tensorboardX, scikit-learn, and tqdm are used for data handling, interpolation, logging, metrics, and progress reporting.

Consult each dependency's distributed license and notices. ImageNet pretrained encoder weights, when requested, are downloaded separately; their use is subject to the upstream terms.

The selected original network and helper files had no project license declaration in the audited source tree. No new license or third-party source provenance is inferred here. This notice is not a substitute for resolving ownership or licensing of code before granting a project-wide license.

## SR experiment lineage

The supplied SR directory's README identifies [UltraSound_SSSR](https://github.com/hengliusky/UltraSound_SSSR) and the following upstream work:

Heng Liu, Jianyong Liu, Shudong Hou, Tao Tao, and Jungong Han. *Perception consistency ultrasound image super-resolution via self-supervised CycleGAN*. Neural Computing and Applications (2021). [DOI](https://doi.org/10.1007/s00521-020-05687-9).

The ZSG publication adapts the user's selected SR experiment. Its entry point and core class have been renamed; unrelated networks/imports, commented-out alternatives, GPU polling, and machine-specific paths have been removed. Device handling and current-library compatibility have been updated. Renaming does not remove the upstream method attribution or imply that all constituent routines originated with ZSG-Net.

## Adapted BSRGAN routines

`sr_degradation.py` and `sr_image.py` contain selected routines adapted through the supplied source from [BSRGAN](https://github.com/cszn/BSRGAN), including the degradation process and interpolation/image-conversion utilities. Copyright 2022 Kai Zhang (https://cszn.github.io/). All rights reserved. These portions are distributed under the Apache License, Version 2.0; the upstream license text is included verbatim in `LICENSE-BSRGAN.txt`.

The modules carry prominent modification notices. Changes include extracting the active dependency closure, removing unrelated/commented-out code, replacing removed SciPy calls, and correcting rectangular mod-crop indexing. Source-local attributions in retained docstrings are preserved. Consult the BSRGAN paper *Designing a Practical Degradation Model for Deep Blind Image Super-Resolution* for its degradation-model lineage.
