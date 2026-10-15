"""Runtime configuration for the supplied fixed-geometry fourfold SR experiment."""
from dataclasses import dataclass, field


@dataclass
class Config:
    input_path: str = ''
    result_path: str = 'results/super_resolution'
    device: str = 'auto'
    encoder_weights: str | None = 'imagenet'
    vgg_weights: str | None = 'imagenet'
    scale_factors: list = field(default_factory=lambda: [[4.0, 4.0]])
    base_change_sfs: list = field(default_factory=list)
    back_projection_iters: list = field(default_factory=lambda: [10])
    max_iters: int = 3000
    min_iters: int = 256
    learning_rate: float = 0.001
    min_learning_rate: float = 9e-6
    learning_rate_change_ratio: float = 1.5
    learning_rate_policy_check_every: int = 60
    learning_rate_slope_range: int = 256
    output_flip: bool = True
    downscale_method: str = 'cubic'
    upscale_method: str = 'cubic'
    downscale_gt_method: str = 'cubic'
    noise_std: float = 0.0
    run_test: bool = True
    run_test_every: int = 60
    display_every: int = 50
    display_test_results: bool = True
    save_results: bool = True
    save_checkpoint: bool = False
    name: str = 'image'
    seed: int = 6666
