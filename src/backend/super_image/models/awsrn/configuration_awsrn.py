from ...configuration_utils import PretrainedConfig
from ...data.datasets import (
    DIV2K_RGB_MEAN,
    DIV2K_RGB_STD
)


class AwsrnConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~super_image.AwsrnModel`.
    It is used to instantiate the model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar
    configuration to that of the [AWSRN](https://huggingface.co/eugenesiow/carn) architecture.
    Configuration objects inherit from :class:`~super_image.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~super_image.PretrainedConfig` for more information.
    Examples:
        ```python
        from super_image import AwsrnModel, AwsrnConfig
        # Initializing a configuration
        config = AwsrnConfig(
            scale=4,                                # train a model to upscale 4x
        )
        # Initializing a model from the configuration
        model = AwsrnModel(config)
        # Accessing the model configuration
        configuration = model.config
        ```
    """
    model_type = 'AWSRN'

    def __init__(self, scale: int = None, n_resblocks=4, block_feats=128, n_colors=3, n_feats=32, res_scale=1, n_awru=4,
                 bam=False, rgb_mean=DIV2K_RGB_MEAN, data_parallel=False, **kwargs):
        """
        Args:
            scale (int): Scale for the model to train an upscaler/super-res model.
            n_resblocks (int): Number of LFB blocks.
            block_feats (int): Number of block features.
            n_feats (int): Number of feature maps.
            n_awru (int): Number of n_awru in one LFB.
            n_colors (int):
                Number of color channels.
            res_scale (int):
                The residual scaling.
            bam (bool): Train using balanced attention.
            rgb_mean (tuple):
                The RGB mean of the train dataset.
                You can use `~super_image.utils.metrics.calculate_mean_std` to calculate it.
            data_parallel (bool):
                Option to use multiple GPUs for training.
        """
        super().__init__(**kwargs)
        self.scale = scale
        self.n_resblocks = n_resblocks
        self.block_feats = block_feats
        self.n_colors = n_colors
        self.n_feats = n_feats
        self.n_awru = n_awru
        self.res_scale = res_scale
        self.bam = bam
        self.rgb_mean = rgb_mean
        self.data_parallel = data_parallel
