from ...configuration_utils import PretrainedConfig
from ...data.datasets import (
    DIV2K_RGB_MEAN,
    DIV2K_RGB_STD
)


class DdbpnConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~super_image.RcanModel`.
    It is used to instantiate the model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar
    configuration to that of the [DDBPN](https://huggingface.co/eugenesiow/ddbpn) architecture.
    Configuration objects inherit from :class:`~super_image.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~super_image.PretrainedConfig` for more information.
    Examples:
        ```python
        from super_image import DdbpnModel, DdbpnConfig
        # Initializing a configuration
        config = DdbpnConfig(
            scale=4,                                # train a model to upscale 4x
        )
        # Initializing a model from the configuration
        model = DdbpnModel(config)
        # Accessing the model configuration
        configuration = model.config
        ```
    """
    model_type = 'DDBPN'

    def __init__(self, scale: int = None, n_colors=3, rgb_range=255, rgb_mean=DIV2K_RGB_MEAN,
                 rgb_std=DIV2K_RGB_STD, data_parallel=False, **kwargs):
        """
        Args:
            scale (int): Scale for the model to train an upscaler/super-res model.
            n_colors (int):
                Number of color channels.
            rgb_range (int):
                Range of RGB as a multiplier to the MeanShift.
            rgb_mean (tuple):
                The RGB mean of the train dataset.
                You can use `~super_image.utils.metrics.calculate_mean_std` to calculate it.
            rgb_std (tuple):
                The RGB standard deviation of the train dataset.
                You can use `~super_image.utils.metrics.calculate_mean_std` to calculate it.
            data_parallel (bool):
                Option to use multiple GPUs for training.
        """
        super().__init__(**kwargs)
        self.scale = scale
        self.n_colors = n_colors
        self.rgb_range = rgb_range
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.data_parallel = data_parallel
