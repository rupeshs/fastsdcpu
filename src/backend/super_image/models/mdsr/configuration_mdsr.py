from ...configuration_utils import PretrainedConfig
from ...data.datasets import (
    DIV2K_RGB_MEAN,
    DIV2K_RGB_STD
)


class MdsrConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~super_image.MdsrModel`.
    It is used to instantiate the model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar
    configuration to that of the [MDSR](https://huggingface.co/eugenesiow/mdsr) architecture.
    Configuration objects inherit from :class:`~super_image.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~super_image.PretrainedConfig` for more information.
    Examples:
        ```python
        from super_image import MdsrModel, MdsrConfig
        # Initializing a configuration
        config = MdsrConfig(
            scale=4,                                # train a model to upscale 4x
        )
        # Initializing a model from the configuration
        model = MdsrModel(config)
        # Accessing the model configuration
        configuration = model.config
        ```
    """
    model_type = 'MDSR'

    def __init__(self, scale: int = None, bam=False, data_parallel=False, **kwargs):
        """
        Args:
            scale (int): Scale for the model to train an upscaler/super-res model.
            data_parallel (bool):
                Option to use multiple GPUs for training.
        """
        super().__init__(**kwargs)
        self.scale = scale
        self.bam = bam
        self.data_parallel = data_parallel
