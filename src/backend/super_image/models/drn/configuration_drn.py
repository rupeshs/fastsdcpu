from ...configuration_utils import PretrainedConfig
import numpy as np
from ...data.datasets import (
    DIV2K_RGB_MEAN,
    DIV2K_RGB_STD
)


class DrnConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~super_image.DrnModel`.
    It is used to instantiate the model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar
    configuration to that of the [DRN](https://huggingface.co/eugenesiow/drn) architecture.
    Configuration objects inherit from :class:`~super_image.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~super_image.PretrainedConfig` for more information.
    Examples:
        ```python
        from super_image import DrnModel, DrnConfig
        # Initializing a configuration
        config = DrnConfig(
            scale=4,                                # train a model to upscale 4x
        )
        # Initializing a model from the configuration
        model = DrnModel(config)
        # Accessing the model configuration
        configuration = model.config
        ```
    """
    model_type = 'DRN'

    def __init__(self, scale: int = 4, n_blocks=16, n_feats=16, negval=0.2, rgb_range=255, data_parallel=False,
                 rgb_mean=DIV2K_RGB_MEAN, rgb_std=DIV2K_RGB_STD, n_colors=3, **kwargs):
        """
        Args:
            scale (int): Scale for the model to train an upscaler/super-res model.
            n_blocks (int): Number of residual blocks, 16|30|40|80.
            n_feats (int): Number of feature maps.
            negval (int): Negative value parameter for Leaky ReLU.
            n_colors (int): Number of color channels to use.
            rgb_mean (tuple):
                The RGB mean of the train dataset.
                You can use `~super_image.utils.metrics.calculate_mean_std` to calculate it.
            rgb_std (tuple):
                The RGB standard deviation of the train dataset.
                You can use `~super_image.utils.metrics.calculate_mean_std` to calculate it.
            rgb_range (int): Maximum value of RGB.
            data_parallel (bool):
                Option to use multiple GPUs for training.
        """
        super().__init__(**kwargs)
        self.scale = [pow(2, s+1) for s in range(int(np.log2(scale)))]
        self.n_blocks = n_blocks
        self.n_feats = n_feats
        self.n_colors = n_colors
        self.negval = negval
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.rgb_range = rgb_range
        self.data_parallel = data_parallel
