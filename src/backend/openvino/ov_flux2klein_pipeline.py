import os
import types
from pathlib import Path
from typing import Optional

import torch
from diffusers import Flux2KleinPipeline
from huggingface_hub import hf_hub_download
from optimum.intel.openvino.modeling_diffusion import OVDiffusionPipeline


def _reshape_ov_part(part, input_shapes: dict) -> None:
    """Reshape an OVPipelinePart model to new input shapes and invalidate its compiled request.

    Args:
        part: An OVPipelinePart instance (e.g. OVModelVaeEncoder / OVModelVaeDecoder).
        input_shapes: Mapping from input tensor name to the desired shape tuple.
    """
    shapes = {}
    for inp in part.model.inputs:
        ps = inp.get_partial_shape()
        name = inp.get_any_name()
        if name in input_shapes:
            for dim_idx, dim_val in enumerate(input_shapes[name]):
                ps[dim_idx] = dim_val
        shapes[inp] = ps
    part.model.reshape(shapes)
    part.request = None  # force recompile on next forward call


class OVFlux2KleinPipeline(OVDiffusionPipeline, Flux2KleinPipeline):
    main_input_name = "prompt"
    export_feature = "text-to-image"
    auto_model_class = Flux2KleinPipeline

    # Layers baked into the exported text encoder graph by Flux2KleinTextEncoderModelPatcher.
    # If the caller requests a different selection we must reject rather than silently return
    # embeddings from the wrong layers.
    _baked_text_encoder_out_layers = (9, 18, 27)

    @classmethod
    def _get_qwen3_prompt_embeds(
        cls,
        text_encoder,
        tokenizer,
        prompt,
        dtype=None,
        device=None,
        max_sequence_length=512,
        hidden_states_layers=(9, 18, 27),
    ):
        """Override to work with OV text encoder that directly outputs stacked prompt_embeds."""
        if tuple(hidden_states_layers) != cls._baked_text_encoder_out_layers:
            raise ValueError(
                f"OVFlux2KleinPipeline was exported with hidden_states_layers="
                f"{cls._baked_text_encoder_out_layers}, but {tuple(hidden_states_layers)} was requested. "
                "Re-export the model with matching text_encoder_out_layers to change this selection."
            )

        prompt = [prompt] if isinstance(prompt, str) else prompt

        all_input_ids = []
        all_attention_masks = []

        for single_prompt in prompt:
            messages = [{"role": "user", "content": single_prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
            )
            all_input_ids.append(inputs["input_ids"])
            all_attention_masks.append(inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0)
        attention_mask = torch.cat(all_attention_masks, dim=0)

        # OV text encoder directly returns prompt_embeds (already stacked from baked-in layers)
        prompt_embeds = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(prompt_embeds, dict):
            prompt_embeds = prompt_embeds["prompt_embeds"]
        elif hasattr(prompt_embeds, "prompt_embeds"):
            prompt_embeds = prompt_embeds.prompt_embeds

        prompt_embeds = (
            torch.from_numpy(prompt_embeds)
            if not isinstance(prompt_embeds, torch.Tensor)
            else prompt_embeds
        )
        if dtype is None:
            dtype = getattr(text_encoder, "dtype", torch.bfloat16)
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds

    def _enc_reshape_to_image(self, image: torch.Tensor) -> None:
        """Reshape and recompile the OV VAE encoder if the image tensor shape changed.

        Named to avoid collision with OVDiffusionPipeline._reshape_vae_encoder, which
        takes a completely different signature (model, batch_size, height, width, ...).
        """
        enc = self.vae.encoder
        shape = tuple(image.shape)
        if getattr(enc, "_compiled_shape", None) == shape:
            return
        print(f"Reshape and compile VAE encoder: {getattr(enc, '_compiled_shape', None)} -> {shape}")
        _reshape_ov_part(enc, {"sample": shape})
        enc._compiled_shape = shape

    def _dec_reshape_to_latents(self, latent_sample: torch.Tensor) -> None:
        """Reshape and recompile the OV VAE decoder if the latent tensor shape changed.

        Named to avoid collision with OVDiffusionPipeline._reshape_vae_decoder, which
        takes a completely different signature (model, height, width, ...).
        """
        dec = self.vae.decoder
        shape = tuple(latent_sample.shape)
        if getattr(dec, "_compiled_shape", None) == shape:
            return
        print(f"Reshape and compile VAE decoder: {getattr(dec, '_compiled_shape', None)} -> {shape}")
        _reshape_ov_part(dec, {"latent_sample": shape})
        dec._compiled_shape = shape

    def _encode_vae_image(self, image: torch.Tensor, generator: Optional[torch.Generator] = None):
        enc = self.vae.encoder
        for inp in enc.model.inputs:
            if inp.get_any_name() == "sample":
                ps = inp.get_partial_shape()
                if ps[2].is_static and ps[3].is_static:
                    th, tw = ps[2].get_length(), ps[3].get_length()
                    if image.shape[2] != th or image.shape[3] != tw:
                        image = torch.nn.functional.interpolate(
                            image.float(), size=(th, tw), mode="bilinear", align_corners=False
                        ).to(image.dtype)
                break
        self._enc_reshape_to_image(image)
        return super()._encode_vae_image(image=image, generator=generator)

    def _reshape_transformer(
        self,
        model,
        batch_size=-1,
        height=-1,
        width=-1,
        num_images_per_prompt=-1,
        tokenizer_max_length=-1,
        num_frames=-1,
    ):
        """Override to use 4-dim IDs (axes_dims_rope=[32,32,32,32]) instead of 3."""
        if batch_size == -1 or num_images_per_prompt == -1:
            batch_size = -1
        else:
            batch_size *= num_images_per_prompt

        # Flux2 packs each 2x2 block of latents into a single transformer token
        # (this is done in the pipeline's _pack_latents, which is also why
        # in_channels == 32 latent channels * 4 == 128). The packed sequence
        # length is therefore (H / vae_scale_factor / 2) * (W / vae_scale_factor / 2),
        # NOT (H / vae_scale_factor) * (W / vae_scale_factor). Forgetting the
        # /2 patch factor bakes a sequence length 4x too large and makes OV
        # reject the real input tensor at inference time.
        patch_size = 2
        height = height // (self.vae_scale_factor * patch_size) if height > 0 else height
        width = width // (self.vae_scale_factor * patch_size) if width > 0 else width
        packed_height_width = width * height if height > 0 and width > 0 else -1

        shapes = {}
        for inputs in model.inputs:
            shapes[inputs] = inputs.get_partial_shape()
            if inputs.get_any_name() in ["timestep", "guidance"]:
                shapes[inputs][0] = batch_size
            elif inputs.get_any_name() == "hidden_states":
                in_channels = self.transformer.config.get("in_channels", 128)
                # Use -1 (dynamic) for the sequence dim: image editing concatenates
                # noise latents + reference image latents whose combined length is
                # not known at reshape time (depends on the reference image size).
                shapes[inputs] = [batch_size, -1, in_channels]
            elif inputs.get_any_name() == "img_ids":
                # Model was exported without the batch dim: shape is [seq, 4].
                # Use -1 so that both noise-only (text-to-image) and
                # noise+reference (image editing) token counts are accepted.
                shapes[inputs] = [-1, 4]
            elif inputs.get_any_name() == "txt_ids":
                # Model was exported without the batch dim: shape is [seq, 4]
                shapes[inputs] = [-1, 4]
            elif inputs.get_any_name() == "encoder_hidden_states":
                shapes[inputs][0] = batch_size
                shapes[inputs][1] = -1
            else:
                shapes[inputs][0] = batch_size
        model.reshape(shapes)
        return model

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        # optimum's from_pretrained does not forward local_files_only to
        # TasksManager.get_model_files, so it always hits hf_api.list_repo_files.
        # For offline/local mode only: resolve the repo ID to its snapshot dir so
        # optimum sees a real directory and uses os.walk instead of the network.
        # For online downloads (local_files_only=False) we let super() handle
        # everything — it downloads each component via its own mechanism.
        if kwargs.get("local_files_only") and not os.path.isdir(str(model_id)):
            from huggingface_hub import snapshot_download
            model_id = snapshot_download(
                repo_id=str(model_id),
                local_files_only=True,
                cache_dir=kwargs.get("cache_dir"),
                revision=kwargs.get("revision"),
                token=kwargs.get("token") or kwargs.get("use_auth_token"),
            )
        return super().from_pretrained(model_id, **kwargs)

    @classmethod
    def _from_pretrained(cls, model_id, config, **kwargs):
        pipeline = super()._from_pretrained(model_id, config, **kwargs)

        # Restore VAE BN running stats + config saved at export time so the parent pipeline
        # can denormalize latents outside of the VAE forward pass. Defaults mirror the
        # upstream AutoencoderKLFlux2 signature (batch_norm_eps=1e-4, block_out_channels=(128,256,512,512)).
        # model_path = Path(model_id) if os.path.isdir(str(model_id)) else model_id
        bn_filename = "vae_bn_stats.npz"

        if os.path.isdir(str(model_id)):
            bn_stats_path = Path(model_id) / bn_filename
        else:
            # Remote repo: download the file from the Hub into the cache
            bn_stats_path = hf_hub_download(
                repo_id=str(model_id),
                filename=bn_filename,
                # forward the auth/cache kwargs the parent uses
                revision=kwargs.get("revision"),
                cache_dir=kwargs.get("cache_dir"),
                token=kwargs.get("token") or kwargs.get("use_auth_token"),
                subfolder=kwargs.get("subfolder"),
                local_files_only=kwargs.get("local_files_only", False),
            )

        bn_stats_path = Path(bn_stats_path)
        print(f"Loading VAE bn stats path {bn_stats_path}")

        batch_norm_eps = None
        block_out_channels = None

        if bn_stats_path.exists():
            import numpy as np

            bn_stats = np.load(bn_stats_path)
            pipeline.vae.bn = types.SimpleNamespace(
                running_mean=torch.from_numpy(bn_stats["running_mean"].copy()),
                running_var=torch.from_numpy(bn_stats["running_var"].copy()),
            )
            if "batch_norm_eps" in bn_stats.files:
                batch_norm_eps = float(bn_stats["batch_norm_eps"])
            if "block_out_channels" in bn_stats.files:
                block_out_channels = list(
                    map(int, bn_stats["block_out_channels"].tolist())
                )

        if not hasattr(pipeline.vae, "config"):
            pipeline.vae.config = types.SimpleNamespace()
        pipeline.vae.config.batch_norm_eps = (
            batch_norm_eps if batch_norm_eps is not None else 1e-4
        )
        pipeline.vae.config.block_out_channels = (
            block_out_channels
            if block_out_channels is not None
            else [128, 256, 512, 512]
        )

        # Flux2KleinPipeline produces img_ids/txt_ids with shape (batch, seq, 4) but
        # the OV model was exported without the batch dim — expects (seq, 4).
        # Patch the transformer forward to squeeze that dim before OV inference.
        _orig_forward = pipeline.transformer.forward

        def _squeeze_ids_forward(*args, **kwargs):
            for key in ("img_ids", "txt_ids"):
                v = kwargs.get(key)
                if isinstance(v, torch.Tensor) and v.ndim == 3:
                    kwargs[key] = v[0]
            return _orig_forward(*args, **kwargs)

        pipeline.transformer.forward = _squeeze_ids_forward

        # Patch VAE decoder forward to reshape for dynamic latent dimensions.
        # There is no _decode_vae_latents hook to override, so we patch here.
        _vae_dec = pipeline.vae.decoder
        _orig_dec_fwd = _vae_dec.forward

        def _dynamic_dec_fwd(latent_sample, **kwargs):
            pipeline._dec_reshape_to_latents(latent_sample)
            return _orig_dec_fwd(latent_sample, **kwargs)

        _vae_dec.forward = _dynamic_dec_fwd

        return pipeline
