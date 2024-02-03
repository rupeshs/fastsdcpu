from ...configuration_utils import PretrainedConfig


MLP_DIM_DEFAULT = [256, 256, 256, 256]


class LiifConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~super_image.JiifModel`.
    It is used to instantiate the model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar
    configuration to that of the [JIIF](https://huggingface.co/eugenesiow/jiif) architecture.
    Configuration objects inherit from :class:`~super_image.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~super_image.PretrainedConfig` for more information.
    Examples:
        ```python
        from super_image import JiifModel, JiifConfig
        # Initializing a configuration
        config = JiifConfig(
            scale=4,                                # train a model to upscale 4x
        )
        # Initializing a model from the configuration
        model = JiifModel(config)
        # Accessing the model configuration
        configuration = model.config
        ```
    """
    model_type = 'LIIF'

    def __init__(self, scale: int = None, local_ensemble=True, feat_unfold=True, cell_decode=True, mlp_dim=None,
                 data_parallel=False, **kwargs):
        """
        Args:
            scale (int): Scale for the model to train an upscaler/super-res model.
            n_resblocks (int): Number of residual blocks.
            data_parallel (bool):
                Option to use multiple GPUs for training.
        """
        super().__init__(**kwargs)
        if mlp_dim is None:
            mlp_dim = MLP_DIM_DEFAULT
        self.scale = scale
        self.mlp_dim = mlp_dim
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.data_parallel = data_parallel
