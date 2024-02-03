from ...configuration_utils import PretrainedConfig
from ...data.datasets import (
    DIV2K_RGB_MEAN,
    DIV2K_RGB_STD
)


class DrlnConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~super_image.DrlnModel`.
    It is used to instantiate the model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar
    configuration to that of the [DRLN](https://huggingface.co/eugenesiow/drln-bam) architecture.
    Configuration objects inherit from :class:`~super_image.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~super_image.PretrainedConfig` for more information.
    Examples:
        ```python
        from super_image import DrlnModel, DrlnConfig
        # Initializing a configuration
        config = DrlnConfig(
            scale=4,                                # train a model to upscale 4x
        )
        # Initializing a model from the configuration
        model = DrlnModel(config)
        # Accessing the model configuration
        configuration = model.config
        ```
    """
    model_type = 'DRLN'

    def __init__(self, scale: int = None, bam=False, data_parallel=False, **kwargs):
        """
        Args:
            scale (int): Scale for the model to train an upscaler/super-res model.
            n_resblocks (int): Number of residual blocks.
            n_feats (int): Number of filters.
            n_colors (int):
                Number of color channels.
            rgb_range (int):
                Range of RGB as a multiplier to the MeanShift.
            res_scale (int):
                The res scale multiplier.
            rgb_mean (tuple):
                The RGB mean of the train dataset.
                You can use `~super_image.utils.metrics.calculate_mean_std` to calculate it.
            rgb_std (tuple):
                The RGB standard deviation of the train dataset.
                You can use `~super_image.utils.metrics.calculate_mean_std` to calculate it.
            no_upsampling (bool):
                Option to turn off upsampling.
            data_parallel (bool):
                Option to use multiple GPUs for training.
        """
        super().__init__(**kwargs)
        self.scale = scale
        self.data_parallel = data_parallel
        self.bam = bam
