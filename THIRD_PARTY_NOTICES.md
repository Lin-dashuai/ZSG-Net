# Third-party components

The release imports third-party libraries through the Python environment; it does not vendor their source or weights.

- [PyTorch](https://github.com/pytorch/pytorch) and [torchvision](https://github.com/pytorch/vision): tensor operations, optimization, and image transformations.
- [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch): EfficientNet-B3 encoder factory.
- [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch): encoder implementation used by the pinned SMP release.
- [timm](https://github.com/huggingface/pytorch-image-models) and [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch): dependencies of the pinned SMP release.
- NumPy, OpenCV, Pillow, tensorboardX, scikit-learn, and tqdm are used for data handling, logging, metrics, and progress reporting.

Consult each dependency's distributed license and notices. ImageNet pretrained encoder weights, when requested, are downloaded separately; their use is subject to the upstream terms.

The selected original network and helper files had no project license declaration in the audited source tree. No new license or third-party source provenance is inferred here. This notice is not a substitute for resolving ownership or licensing of code before granting a project-wide license.
