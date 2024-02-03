from ...configuration_utils import PretrainedConfig
from ...data.datasets import (
    DIV2K_RGB_MEAN,
    DIV2K_RGB_STD
)


class MsrnConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~super_image.MsrnModel`.
    It is used to instantiate the model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar
    configuration to that of the [MSRN BAM](https://huggingface.co/eugenesiow/msrn-bam) architecture.
    Configuration objects inherit from :class:`~super_image.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~super_image.PretrainedConfig` for more information.
    Examples:
        ```python
        from super_image import MsrnModel, MsrnConfig
        # Initializing a configuration
        config = MsrnConfig(
            scale=4,                                # train a model to upscale 4x
            bam=True,                               # use balanced attention (BAM)
        )
        # Initializing a model from the configuration
        model = MsrnModel(config)
        # Accessing the model configuration
        configuration = model.config
        ```
    """
    model_type = 'MSRN'

    def __init__(self, scale=None, n_blocks=8, n_feats=64, rgb_range=255, bam=False,
                 rgb_mean=DIV2K_RGB_MEAN, rgb_std=DIV2K_RGB_STD,
                 data_parallel=False, **kwargs):
        """
        Args:
            scale (int): Scale for the model to train an upscaler/super-res model.
            n_blocks (int): Number of blocks.
            n_feats (int): Number of filters.
            rgb_range (int):
                Range of RGB as a multiplier to the MeanShift.
            data_parallel (bool):
                Option to use multiple GPUs for training.
            bam (bool): Option to use balanced attention modules instead (BAM)
        """
        super().__init__(**kwargs)
        self.scale = scale
        self.n_blocks = n_blocks
        self.n_feats = n_feats
        self.rgb_range = rgb_range
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.data_parallel = data_parallel
        self.bam = bam
