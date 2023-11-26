from backend.device import is_openvino_device

if is_openvino_device():
    from optimum.intel.openvino.modeling_diffusion import OVModelVaeDecoder


class CustomOVModelVaeDecoder(OVModelVaeDecoder):
    def __init__(
        self,
        model,
        parent_model,
        ov_config=None,
        model_dir=None,
    ):
        super(OVModelVaeDecoder, self).__init__(
            model,
            parent_model,
            ov_config,
            "vae_decoder",
            model_dir,
        )
