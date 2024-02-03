from ...configuration_utils import PretrainedConfig


class MasaConfig(PretrainedConfig):
    model_type = 'MASA'

    def __init__(self, scale=None, input_nc=3, nf=64, num_nbr=1, data_parallel=False, **kwargs):
        super().__init__(**kwargs)
        self.num_nbr = num_nbr
        self.input_nc = input_nc
        self.nf = nf
        self.scale = scale
        self.data_parallel = data_parallel
