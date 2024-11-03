"""
Copyright(C) 2022-2023 Intel Corporation
SPDX - License - Identifier: Apache - 2.0

"""
import inspect
from typing import Union, Optional, Any, List, Dict
import numpy as np
# openvino
from openvino.runtime import Core
# tokenizer
from transformers import CLIPTokenizer
import torch
import random

from diffusers import DiffusionPipeline
from diffusers.schedulers import (DDIMScheduler,
                                  LMSDiscreteScheduler,
                                  PNDMScheduler,
                                  EulerDiscreteScheduler,
                                  EulerAncestralDiscreteScheduler)


from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import PIL_INTERPOLATION

import cv2
import os
import sys

# for multithreading 
import concurrent.futures

#For GIF
import PIL
from PIL import Image
import glob
import json
import time

def scale_fit_to_window(dst_width:int, dst_height:int, image_width:int, image_height:int):
    """
    Preprocessing helper function for calculating image size for resize with peserving original aspect ratio
    and fitting image to specific window size

    Parameters:
      dst_width (int): destination window width
      dst_height (int): destination window height
      image_width (int): source image width
      image_height (int): source image height
    Returns:
      result_width (int): calculated width for resize
      result_height (int): calculated height for resize
    """
    im_scale = min(dst_height / image_height, dst_width / image_width)
    return int(im_scale * image_width), int(im_scale * image_height)

def preprocess(image: PIL.Image.Image, ht=512, wt=512):
    """
    Image preprocessing function. Takes image in PIL.Image format, resizes it to keep aspect ration and fits to model input window 512x512,
    then converts it to np.ndarray and adds padding with zeros on right or bottom side of image (depends from aspect ratio), after that
    converts data to float32 data type and change range of values from [0, 255] to [-1, 1], finally, converts data layout from planar NHWC to NCHW.
    The function returns preprocessed input tensor and padding size, which can be used in postprocessing.

    Parameters:
      image (PIL.Image.Image): input image
    Returns:
       image (np.ndarray): preprocessed image tensor
       meta (Dict): dictionary with preprocessing metadata info
    """

    src_width, src_height = image.size
    image = image.convert('RGB')
    dst_width, dst_height = scale_fit_to_window(
        wt, ht, src_width, src_height)
    image = np.array(image.resize((dst_width, dst_height),
                     resample=PIL.Image.Resampling.LANCZOS))[None, :]

    pad_width = wt - dst_width
    pad_height = ht - dst_height
    pad = ((0, 0), (0, pad_height), (0, pad_width), (0, 0))
    image = np.pad(image, pad, mode="constant")
    image = image.astype(np.float32) / 255.0
    image = 2.0 * image - 1.0
    image = image.transpose(0, 3, 1, 2)

    return image, {"padding": pad, "src_width": src_width, "src_height": src_height}

def try_enable_npu_turbo(device, core):
    import platform
    if "windows" in platform.system().lower():
        if "NPU" in device and "3720" not in core.get_property('NPU', 'DEVICE_ARCHITECTURE'):
            try:
                core.set_property(properties={'NPU_TURBO': 'YES'},device_name='NPU')
            except:
                print(f"Failed loading NPU_TURBO for device {device}. Skipping... ")
            else:
                print_npu_turbo_art()
        else:
            print(f"Skipping NPU_TURBO for device {device}")
    elif "linux" in platform.system().lower():
        if os.path.isfile('/sys/module/intel_vpu/parameters/test_mode'):
            with open('/sys/module/intel_vpu/version', 'r') as f:
                version = f.readline().split()[0]
                if tuple(map(int, version.split('.'))) < tuple(map(int, '1.9.0'.split('.'))):
                    print(f"The driver intel_vpu-1.9.0 (or later) needs to be loaded for NPU Turbo (currently {version}). Skipping...")
                else:
                    with open('/sys/module/intel_vpu/parameters/test_mode', 'r') as tm_file:
                        test_mode = int(tm_file.readline().split()[0])
                        if test_mode == 512:
                            print_npu_turbo_art()
                        else:
                            print("The driver >=intel_vpu-1.9.0 was must be loaded with "
                                  "\"modprobe intel_vpu test_mode=512\" to enable NPU_TURBO "
                                  f"(currently test_mode={test_mode}). Skipping...")
        else:
            print(f"The driver >=intel_vpu-1.9.0 must be loaded with  \"modprobe intel_vpu test_mode=512\" to enable NPU_TURBO. Skipping...")
    else:
        print(f"This platform ({platform.system()}) does not support NPU Turbo")

def result(var):
    return next(iter(var.values()))

class StableDiffusionEngineAdvanced(DiffusionPipeline):
    def __init__(self, model="runwayml/stable-diffusion-v1-5", 
                  tokenizer="openai/clip-vit-large-patch14", 
                  device=["CPU", "CPU", "CPU", "CPU"]):
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(model, local_files_only=True)
        except:
            self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer)
            self.tokenizer.save_pretrained(model)

        self.core = Core()
        self.core.set_property({'CACHE_DIR': os.path.join(model, 'cache')})
        try_enable_npu_turbo(device, self.core)
            
        print("Loading models... ")
        


        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                "unet_time_proj": executor.submit(self.core.compile_model, os.path.join(model, "unet_time_proj.xml"), device[0]),
                "text": executor.submit(self.load_model, model, "text_encoder", device[0]),
                "unet": executor.submit(self.load_model, model, "unet_int8", device[1]),
                "unet_neg": executor.submit(self.load_model, model, "unet_int8", device[2]) if device[1] != device[2] else None,
                "vae_decoder": executor.submit(self.load_model, model, "vae_decoder", device[3]),
                "vae_encoder": executor.submit(self.load_model, model, "vae_encoder", device[3])
            }

        self.unet_time_proj = futures["unet_time_proj"].result()
        self.text_encoder = futures["text"].result()
        self.unet = futures["unet"].result()
        self.unet_neg = futures["unet_neg"].result() if futures["unet_neg"] else self.unet
        self.vae_decoder = futures["vae_decoder"].result()
        self.vae_encoder = futures["vae_encoder"].result()
        print("Text Device:", device[0])
        print("unet Device:", device[1])
        print("unet-neg Device:", device[2])
        print("VAE Device:", device[3])

        self._text_encoder_output = self.text_encoder.output(0)
        self._vae_d_output = self.vae_decoder.output(0)
        self._vae_e_output = self.vae_encoder.output(0) if self.vae_encoder else None

        self.set_dimensions()
        self.infer_request_neg = self.unet_neg.create_infer_request()
        self.infer_request = self.unet.create_infer_request()
        self.infer_request_time_proj = self.unet_time_proj.create_infer_request()
        self.time_proj_constants = np.load(os.path.join(model, "time_proj_constants.npy"))
        
    def load_model(self, model, model_name, device):
        if "NPU" in device:
            with open(os.path.join(model, f"{model_name}.blob"), "rb") as f:
                return self.core.import_model(f.read(), device)
        return self.core.compile_model(os.path.join(model, f"{model_name}.xml"), device)
    
    def set_dimensions(self):
        latent_shape = self.unet.input("latent_model_input").shape
        if latent_shape[1] == 4:
            self.height = latent_shape[2] * 8
            self.width = latent_shape[3] * 8
        else:
            self.height = latent_shape[1] * 8
            self.width = latent_shape[2] * 8

    def __call__(
            self,
            prompt,
            init_image = None,
            negative_prompt=None,
            scheduler=None,
            strength = 0.5,
            num_inference_steps = 32,
            guidance_scale = 7.5,
            eta = 0.0,
            create_gif = False,
            model = None,
            callback = None,
            callback_userdata = None
    ):

        # extract condition
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        text_embeddings = self.text_encoder(text_input.input_ids)[self._text_encoder_output]

        # do classifier free guidance
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:

            if negative_prompt is None:
                uncond_tokens = [""]
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = negative_prompt

            tokens_uncond = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length, #truncation=True,
                return_tensors="np"
            )
            uncond_embeddings = self.text_encoder(tokens_uncond.input_ids)[self._text_encoder_output]
            text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}

        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, scheduler)
        latent_timestep = timesteps[:1]

        # get the initial random noise unless the user supplied it
        latents, meta = self.prepare_latents(init_image, latent_timestep, scheduler)


        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        if create_gif:
            frames = []

        for i, t in enumerate(self.progress_bar(timesteps)):
            if callback:
               callback(i, callback_userdata)

            # expand the latents if we are doing classifier free guidance
            noise_pred = []
            latent_model_input = latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            latent_model_input_neg = latent_model_input
            if self.unet.input("latent_model_input").shape[1] != 4:
                #print("In transpose")
                try:
                    latent_model_input = latent_model_input.permute(0,2,3,1)
                except:
                    latent_model_input = latent_model_input.transpose(0,2,3,1)

            if self.unet_neg.input("latent_model_input").shape[1] != 4:
                #print("In transpose")
                try:
                    latent_model_input_neg = latent_model_input_neg.permute(0,2,3,1)
                except:
                    latent_model_input_neg = latent_model_input_neg.transpose(0,2,3,1)


            time_proj_constants_fp16 = np.float16(self.time_proj_constants)
            t_scaled_fp16 = time_proj_constants_fp16 * np.float16(t)
            cosine_t_fp16 = np.cos(t_scaled_fp16)
            sine_t_fp16 = np.sin(t_scaled_fp16)

            t_scaled = self.time_proj_constants * np.float32(t)

            cosine_t = np.cos(t_scaled)
            sine_t = np.sin(t_scaled)

            time_proj_dict = {"sine_t" : np.float32(sine_t), "cosine_t" : np.float32(cosine_t)}
            self.infer_request_time_proj.start_async(time_proj_dict)
            self.infer_request_time_proj.wait()
            time_proj = self.infer_request_time_proj.get_output_tensor(0).data.astype(np.float32)

            input_tens_neg_dict = {"time_proj": np.float32(time_proj), "latent_model_input":latent_model_input_neg, "encoder_hidden_states": np.expand_dims(text_embeddings[0], axis=0)}
            input_tens_dict = {"time_proj": np.float32(time_proj), "latent_model_input":latent_model_input, "encoder_hidden_states": np.expand_dims(text_embeddings[1], axis=0)}

            self.infer_request_neg.start_async(input_tens_neg_dict)
            self.infer_request.start_async(input_tens_dict)
            self.infer_request_neg.wait()
            self.infer_request.wait()
            
            noise_pred_neg = self.infer_request_neg.get_output_tensor(0)
            noise_pred_pos = self.infer_request.get_output_tensor(0)

            noise_pred.append(noise_pred_neg.data.astype(np.float32))
            noise_pred.append(noise_pred_pos.data.astype(np.float32))

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred[0], noise_pred[1]
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs)["prev_sample"].numpy()

            if create_gif:
                frames.append(latents)

        if callback:
            callback(num_inference_steps, callback_userdata)

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents

        start = time.time()
        image = self.vae_decoder(latents)[self._vae_d_output]
        print("Decoder ended:",time.time() - start)

        image = self.postprocess_image(image, meta)

        if create_gif:
            gif_folder=os.path.join(model,"../../../gif")
            print("gif_folder:",gif_folder)
            if not os.path.exists(gif_folder):
                os.makedirs(gif_folder)
            for i in range(0,len(frames)):
                image = self.vae_decoder(frames[i]*(1/0.18215))[self._vae_d_output]
                image = self.postprocess_image(image, meta)
                output = gif_folder + "/" + str(i).zfill(3) +".png"
                cv2.imwrite(output, image)
            with open(os.path.join(gif_folder, "prompt.json"), "w") as file:
                json.dump({"prompt": prompt}, file)
            frames_image =  [Image.open(image) for image in glob.glob(f"{gif_folder}/*.png")]
            frame_one = frames_image[0]
            gif_file=os.path.join(gif_folder,"stable_diffusion.gif")
            frame_one.save(gif_file, format="GIF", append_images=frames_image, save_all=True, duration=100, loop=0)

        return image

    def prepare_latents(self, image:PIL.Image.Image = None, latent_timestep:torch.Tensor = None, scheduler = LMSDiscreteScheduler):
        """
        Function for getting initial latents for starting generation

        Parameters:
            image (PIL.Image.Image, *optional*, None):
                Input image for generation, if not provided randon noise will be used as starting point
            latent_timestep (torch.Tensor, *optional*, None):
                Predicted by scheduler initial step for image generation, required for latent image mixing with nosie
        Returns:
            latents (np.ndarray):
                Image encoded in latent space
        """
        latents_shape = (1, 4, self.height // 8, self.width // 8)
   
        noise = np.random.randn(*latents_shape).astype(np.float32)
        if image is None:
            ##print("Image is NONE")
            # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
            if isinstance(scheduler, LMSDiscreteScheduler):

                noise = noise * scheduler.sigmas[0].numpy()
                return noise, {}
            elif isinstance(scheduler, EulerDiscreteScheduler) or isinstance(scheduler,EulerAncestralDiscreteScheduler):

                noise = noise * scheduler.sigmas.max().numpy()
                return noise, {}
            else:
                return noise, {}
        input_image, meta = preprocess(image,self.height,self.width)
       
        moments = self.vae_encoder(input_image)[self._vae_e_output]
      
        mean, logvar = np.split(moments, 2, axis=1)
  
        std = np.exp(logvar * 0.5)
        latents = (mean + std * np.random.randn(*mean.shape)) * 0.18215
       
         
        latents = scheduler.add_noise(torch.from_numpy(latents), torch.from_numpy(noise), latent_timestep).numpy()
        return latents, meta

    def postprocess_image(self, image:np.ndarray, meta:Dict):
        """
        Postprocessing for decoded image. Takes generated image decoded by VAE decoder, unpad it to initial image size (if required), 
        normalize and convert to [0, 255] pixels range. Optionally, convertes it from np.ndarray to PIL.Image format

        Parameters:
            image (np.ndarray):
                Generated image
            meta (Dict):
                Metadata obtained on latents preparing step, can be empty
            output_type (str, *optional*, pil):
                Output format for result, can be pil or numpy
        Returns:
            image (List of np.ndarray or PIL.Image.Image):
                Postprocessed images

                        if "src_height" in meta:
            orig_height, orig_width = meta["src_height"], meta["src_width"]
            image = [cv2.resize(img, (orig_width, orig_height))
                        for img in image]

        return image
        """
        if "padding" in meta:
            pad = meta["padding"]
            (_, end_h), (_, end_w) = pad[1:3]
            h, w = image.shape[2:]
            #print("image shape",image.shape[2:])
            unpad_h = h - end_h
            unpad_w = w - end_w
            image = image[:, :, :unpad_h, :unpad_w]
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = (image[0].transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)



        if "src_height" in meta:
            orig_height, orig_width = meta["src_height"], meta["src_width"]
            image = cv2.resize(image, (orig_width, orig_height))

        return image




    def get_timesteps(self, num_inference_steps:int, strength:float, scheduler):
        """
        Helper function for getting scheduler timesteps for generation
        In case of image-to-image generation, it updates number of steps according to strength

        Parameters:
           num_inference_steps (int):
              number of inference steps for generation
           strength (float):
               value between 0.0 and 1.0, that controls the amount of noise that is added to the input image.
               Values that approach 1.0 allow for lots of variations but will also produce images that are not semantically consistent with the input.
        """
        # get the original timestep using init_timestep

        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

class StableDiffusionEngine(DiffusionPipeline):
    def __init__(
            self,
            model="bes-dev/stable-diffusion-v1-4-openvino",
            tokenizer="openai/clip-vit-large-patch14",
            device=["CPU","CPU","CPU","CPU"]):
        
        self.core = Core()
        self.core.set_property({'CACHE_DIR': os.path.join(model, 'cache')})

        self.batch_size = 2 if device[1] == device[2] and device[1] == "GPU" else 1
        try_enable_npu_turbo(device, self.core)

        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(model, local_files_only=True)
        except Exception as e:
            print("Local tokenizer not found. Attempting to download...")
            self.tokenizer = self.download_tokenizer(tokenizer, model)
    
        print("Loading models... ")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            text_future = executor.submit(self.load_model, model, "text_encoder", device[0])
            vae_de_future = executor.submit(self.load_model, model, "vae_decoder", device[3])
            vae_en_future = executor.submit(self.load_model, model, "vae_encoder", device[3])

            if self.batch_size == 1:
                if "int8" not in model:
                    unet_future = executor.submit(self.load_model, model, "unet_bs1", device[1])
                    unet_neg_future = executor.submit(self.load_model, model, "unet_bs1", device[2]) if device[1] != device[2] else None
                else:
                    unet_future = executor.submit(self.load_model, model, "unet_int8a16", device[1])
                    unet_neg_future = executor.submit(self.load_model, model, "unet_int8a16", device[2]) if device[1] != device[2] else None
            else:
                unet_future = executor.submit(self.load_model, model, "unet", device[1])
                unet_neg_future = None

            self.unet = unet_future.result()
            self.unet_neg = unet_neg_future.result() if unet_neg_future else self.unet
            self.text_encoder = text_future.result()
            self.vae_decoder = vae_de_future.result()
            self.vae_encoder = vae_en_future.result()
            print("Text Device:", device[0])
            print("unet Device:", device[1])
            print("unet-neg Device:", device[2])
            print("VAE Device:", device[3])

            self._text_encoder_output = self.text_encoder.output(0)
            self._unet_output = self.unet.output(0)
            self._vae_d_output = self.vae_decoder.output(0)
            self._vae_e_output = self.vae_encoder.output(0) if self.vae_encoder else None

            self.unet_input_tensor_name = "sample" if 'sample' in self.unet.input(0).names else "latent_model_input"

            if self.batch_size == 1:
                self.infer_request = self.unet.create_infer_request()
                self.infer_request_neg = self.unet_neg.create_infer_request()
                self._unet_neg_output = self.unet_neg.output(0)
            else:
                self.infer_request = None
                self.infer_request_neg = None
                self._unet_neg_output = None
         
        self.set_dimensions()

        

    def load_model(self, model, model_name, device):
        if "NPU" in device:
            with open(os.path.join(model, f"{model_name}.blob"), "rb") as f:
                return self.core.import_model(f.read(), device)
        return self.core.compile_model(os.path.join(model, f"{model_name}.xml"), device)
        
    def set_dimensions(self):
        latent_shape = self.unet.input(self.unet_input_tensor_name).shape
        if latent_shape[1] == 4:
            self.height = latent_shape[2] * 8
            self.width = latent_shape[3] * 8
        else:
            self.height = latent_shape[1] * 8
            self.width = latent_shape[2] * 8

    def __call__(
            self,
            prompt,
            init_image=None,
            negative_prompt=None,
            scheduler=None,
            strength=0.5,
            num_inference_steps=32,
            guidance_scale=7.5,
            eta=0.0,
            create_gif=False,
            model=None,
            callback=None,
            callback_userdata=None
    ):
        # extract condition
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        text_embeddings = self.text_encoder(text_input.input_ids)[self._text_encoder_output]
        

        # do classifier free guidance
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            if negative_prompt is None:
                uncond_tokens = [""]
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = negative_prompt

            tokens_uncond = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,  # truncation=True,
                return_tensors="np"
            )
            uncond_embeddings = self.text_encoder(tokens_uncond.input_ids)[self._text_encoder_output]
            text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}

        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, scheduler)
        latent_timestep = timesteps[:1]

        # get the initial random noise unless the user supplied it
        latents, meta = self.prepare_latents(init_image, latent_timestep, scheduler,model)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        if create_gif:
            frames = []

        for i, t in enumerate(self.progress_bar(timesteps)):
            if callback:
                callback(i, callback_userdata)

            if self.batch_size == 1:
                # expand the latents if we are doing classifier free guidance
                noise_pred = []
                latent_model_input = latents 
                   
                #Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                latent_model_input_pos = latent_model_input 
                latent_model_input_neg = latent_model_input

                if self.unet.input(self.unet_input_tensor_name).shape[1] != 4:
                    try:
                        latent_model_input_pos = latent_model_input_pos.permute(0,2,3,1)
                    except:
                        latent_model_input_pos = latent_model_input_pos.transpose(0,2,3,1)
                
                if self.unet_neg.input(self.unet_input_tensor_name).shape[1] != 4:
                    try:
                        latent_model_input_neg = latent_model_input_neg.permute(0,2,3,1)
                    except:
                        latent_model_input_neg = latent_model_input_neg.transpose(0,2,3,1)
                
                if "sample" in self.unet_input_tensor_name:                                        
                    input_tens_neg_dict = {"sample" : latent_model_input_neg, "encoder_hidden_states": np.expand_dims(text_embeddings[0], axis=0), "timestep": np.expand_dims(np.float32(t), axis=0)}
                    input_tens_pos_dict = {"sample" : latent_model_input_pos, "encoder_hidden_states": np.expand_dims(text_embeddings[1], axis=0), "timestep": np.expand_dims(np.float32(t), axis=0)}
                else:
                    input_tens_neg_dict = {"latent_model_input" : latent_model_input_neg, "encoder_hidden_states": np.expand_dims(text_embeddings[0], axis=0), "t": np.expand_dims(np.float32(t), axis=0)}
                    input_tens_pos_dict = {"latent_model_input" : latent_model_input_pos, "encoder_hidden_states": np.expand_dims(text_embeddings[1], axis=0), "t": np.expand_dims(np.float32(t), axis=0)}
                                                     
                self.infer_request_neg.start_async(input_tens_neg_dict)
                self.infer_request.start_async(input_tens_pos_dict)    
         
                self.infer_request_neg.wait()
                self.infer_request.wait()

                noise_pred_neg = self.infer_request_neg.get_output_tensor(0)
                noise_pred_pos = self.infer_request.get_output_tensor(0)
                               
                noise_pred.append(noise_pred_neg.data.astype(np.float32))
                noise_pred.append(noise_pred_pos.data.astype(np.float32))
            else:
                latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                noise_pred = self.unet([latent_model_input, np.array(t, dtype=np.float32), text_embeddings])[self._unet_output]
                
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred[0], noise_pred[1]
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs)["prev_sample"].numpy()

            if create_gif:
                frames.append(latents)

        if callback:
            callback(num_inference_steps, callback_userdata)

        # scale and decode the image latents with vae
        #if self.height == 512 and self.width == 512:
        latents = 1 / 0.18215 * latents
        image = self.vae_decoder(latents)[self._vae_d_output]
        image = self.postprocess_image(image, meta)

        return image

    def prepare_latents(self, image: PIL.Image.Image = None, latent_timestep: torch.Tensor = None,
                        scheduler=LMSDiscreteScheduler,model=None):
        """
        Function for getting initial latents for starting generation

        Parameters:
            image (PIL.Image.Image, *optional*, None):
                Input image for generation, if not provided randon noise will be used as starting point
            latent_timestep (torch.Tensor, *optional*, None):
                Predicted by scheduler initial step for image generation, required for latent image mixing with nosie
        Returns:
            latents (np.ndarray):
                Image encoded in latent space
        """
        latents_shape = (1, 4, self.height // 8, self.width // 8)

        noise = np.random.randn(*latents_shape).astype(np.float32)
        if image is None:
            #print("Image is NONE")
            # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
            if isinstance(scheduler, LMSDiscreteScheduler):

                noise = noise * scheduler.sigmas[0].numpy()
                return noise, {}
            elif isinstance(scheduler, EulerDiscreteScheduler):

                noise = noise * scheduler.sigmas.max().numpy()
                return noise, {}
            else:
                return noise, {}
        input_image, meta = preprocess(image, self.height, self.width)

        moments = self.vae_encoder(input_image)[self._vae_e_output]

        if "sd_2.1" in model:
            latents = moments * 0.18215

        else:

            mean, logvar = np.split(moments, 2, axis=1)

            std = np.exp(logvar * 0.5)
            latents = (mean + std * np.random.randn(*mean.shape)) * 0.18215

        latents = scheduler.add_noise(torch.from_numpy(latents), torch.from_numpy(noise), latent_timestep).numpy()
        return latents, meta
        
  
    def postprocess_image(self, image: np.ndarray, meta: Dict):
        """
        Postprocessing for decoded image. Takes generated image decoded by VAE decoder, unpad it to initila image size (if required),
        normalize and convert to [0, 255] pixels range. Optionally, convertes it from np.ndarray to PIL.Image format

        Parameters:
            image (np.ndarray):
                Generated image
            meta (Dict):
                Metadata obtained on latents preparing step, can be empty
            output_type (str, *optional*, pil):
                Output format for result, can be pil or numpy
        Returns:
            image (List of np.ndarray or PIL.Image.Image):
                Postprocessed images

                        if "src_height" in meta:
            orig_height, orig_width = meta["src_height"], meta["src_width"]
            image = [cv2.resize(img, (orig_width, orig_height))
                        for img in image]

        return image
        """
        if "padding" in meta:
            pad = meta["padding"]
            (_, end_h), (_, end_w) = pad[1:3]
            h, w = image.shape[2:]
            # print("image shape",image.shape[2:])
            unpad_h = h - end_h
            unpad_w = w - end_w
            image = image[:, :, :unpad_h, :unpad_w]
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = (image[0].transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)

        if "src_height" in meta:
            orig_height, orig_width = meta["src_height"], meta["src_width"]
            image = cv2.resize(image, (orig_width, orig_height))

        return image

        # image = (image / 2 + 0.5).clip(0, 1)
        # image = (image[0].transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)

    def get_timesteps(self, num_inference_steps: int, strength: float, scheduler):
        """
        Helper function for getting scheduler timesteps for generation
        In case of image-to-image generation, it updates number of steps according to strength

        Parameters:
           num_inference_steps (int):
              number of inference steps for generation
           strength (float):
               value between 0.0 and 1.0, that controls the amount of noise that is added to the input image.
               Values that approach 1.0 allow for lots of variations but will also produce images that are not semantically consistent with the input.
        """
        # get the original timestep using init_timestep

        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

class LatentConsistencyEngine(DiffusionPipeline):
    def __init__(
        self,
            model="SimianLuo/LCM_Dreamshaper_v7",
            tokenizer="openai/clip-vit-large-patch14",
            device=["CPU", "CPU", "CPU"],
    ):
        super().__init__()
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(model, local_files_only=True)
        except:
            self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer)
            self.tokenizer.save_pretrained(model)

        self.core = Core()
        self.core.set_property({'CACHE_DIR': os.path.join(model, 'cache')})  # adding caching to reduce init time
        try_enable_npu_turbo(device, self.core)
               

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            text_future = executor.submit(self.load_model, model, "text_encoder", device[0])
            unet_future = executor.submit(self.load_model, model, "unet", device[1])    
            vae_de_future = executor.submit(self.load_model, model, "vae_decoder", device[2])
                
        print("Text Device:", device[0])
        self.text_encoder = text_future.result()
        self._text_encoder_output = self.text_encoder.output(0)

        print("Unet Device:", device[1])
        self.unet = unet_future.result()
        self._unet_output = self.unet.output(0)
        self.infer_request = self.unet.create_infer_request()

        print(f"VAE Device: {device[2]}")
        self.vae_decoder = vae_de_future.result()
        self.infer_request_vae = self.vae_decoder.create_infer_request()
        self.safety_checker = None #pipe.safety_checker
        self.feature_extractor = None #pipe.feature_extractor
        self.vae_scale_factor = 2 ** 3
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def load_model(self, model, model_name, device):
        if "NPU" in device:
            with open(os.path.join(model, f"{model_name}.blob"), "rb") as f:
                return self.core.import_model(f.read(), device)
        return self.core.compile_model(os.path.join(model, f"{model_name}.xml"), device)

    def _encode_prompt(
        self,
        prompt,
        num_images_per_prompt,
        prompt_embeds: None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
        """

        if prompt_embeds is None:

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_embeds = self.text_encoder(text_input_ids, share_inputs=True, share_outputs=True)
            prompt_embeds = torch.from_numpy(prompt_embeds[0])

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        # Don't need to get uncond prompt embedding because of LCM Guided Distillation
        return prompt_embeds

    def run_safety_checker(self, image, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(
                    image, output_type="pil"
                )
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(
                feature_extractor_input, return_tensors="pt"
            )
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    def prepare_latents(
        self, batch_size, num_channels_latents, height, width, dtype, latents=None
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if latents is None:
            latents = torch.randn(shape, dtype=dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        return latents

    def get_w_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        see https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298
        Args:
        timesteps: torch.Tensor: generate embedding vectors at these timesteps
        embedding_dim: int: dimension of the embeddings to generate
        dtype: data type of the generated embeddings
        Returns:
        embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        guidance_scale: float = 7.5,
        scheduler = None,
        num_images_per_prompt: Optional[int] = 1,
        latents: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 4,
        lcm_origin_steps: int = 50,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        model: Optional[Dict[str, any]] = None,
        seed: Optional[int] = 1234567,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback = None,
        callback_userdata = None
    ):

        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if seed is not None:
            torch.manual_seed(seed)

        #print("After Step 1: batch size is ", batch_size)
        # do_classifier_free_guidance = guidance_scale > 0.0
        # In LCM Implementation:  cfg_noise = noise_cond + cfg_scale * (noise_cond - noise_uncond) , (cfg_scale > 0.0 using CFG)

        # 2. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            prompt_embeds=prompt_embeds,
        )
        #print("After Step 2: prompt embeds is ", prompt_embeds)
        #print("After Step 2: scheduler is ", scheduler )
        # 3. Prepare timesteps
        scheduler.set_timesteps(num_inference_steps, original_inference_steps=lcm_origin_steps)
        timesteps = scheduler.timesteps

        #print("After Step 3: timesteps is ", timesteps)

        # 4. Prepare latent variable
        num_channels_latents = 4
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            latents,
        )
        latents = latents * scheduler.init_noise_sigma

        #print("After Step 4: ")
        bs = batch_size * num_images_per_prompt

        # 5. Get Guidance Scale Embedding
        w = torch.tensor(guidance_scale).repeat(bs)
        w_embedding = self.get_w_embedding(w, embedding_dim=256)
        #print("After Step 5: ")
        # 6. LCM MultiStep Sampling Loop:
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if callback:
                    callback(i+1, callback_userdata)

                ts = torch.full((bs,), t, dtype=torch.long)

                # model prediction (v-prediction, eps, x)
                model_pred = self.unet([latents, ts, prompt_embeds, w_embedding],share_inputs=True, share_outputs=True)[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents, denoised = scheduler.step(
                    torch.from_numpy(model_pred), t, latents, return_dict=False
                )
                progress_bar.update()

        #print("After Step 6: ")

        vae_start = time.time()

        if not output_type == "latent":
            image = torch.from_numpy(self.vae_decoder(denoised / 0.18215, share_inputs=True, share_outputs=True)[0])
        else:
            image = denoised

        print("Decoder Ended: ", time.time() - vae_start)
        #post_start = time.time()

        #if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
        #else:
        #    do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        #print ("After do_denormalize: image is ", image)

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        return image[0]

class LatentConsistencyEngineAdvanced(DiffusionPipeline):
    def __init__(
        self,
            model="SimianLuo/LCM_Dreamshaper_v7",
            tokenizer="openai/clip-vit-large-patch14",
            device=["CPU", "CPU", "CPU"],
    ):
        super().__init__()
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(model, local_files_only=True)
        except:
            self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer)
            self.tokenizer.save_pretrained(model)

        self.core = Core()
        self.core.set_property({'CACHE_DIR': os.path.join(model, 'cache')})  # adding caching to reduce init time
        #try_enable_npu_turbo(device, self.core)
               

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            text_future = executor.submit(self.load_model, model, "text_encoder", device[0])
            unet_future = executor.submit(self.load_model, model, "unet", device[1])    
            vae_de_future = executor.submit(self.load_model, model, "vae_decoder", device[2])
            vae_encoder_future = executor.submit(self.load_model, model, "vae_encoder", device[2])
                
       
        print("Text Device:", device[0])
        self.text_encoder = text_future.result()
        self._text_encoder_output = self.text_encoder.output(0)

        print("Unet Device:", device[1])
        self.unet = unet_future.result()
        self._unet_output = self.unet.output(0)
        self.infer_request = self.unet.create_infer_request()

        print(f"VAE Device: {device[2]}")
        self.vae_decoder = vae_de_future.result()
        self.vae_encoder = vae_encoder_future.result()
        self._vae_e_output = self.vae_encoder.output(0) if self.vae_encoder else None

        self.infer_request_vae = self.vae_decoder.create_infer_request()
        self.safety_checker = None #pipe.safety_checker
        self.feature_extractor = None #pipe.feature_extractor
        self.vae_scale_factor = 2 ** 3
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def load_model(self, model, model_name, device):
        print(f"Compiling the {model_name} to {device} ...")
        return self.core.compile_model(os.path.join(model, f"{model_name}.xml"), device)
    
    def get_timesteps(self, num_inference_steps:int, strength:float, scheduler):
        """
        Helper function for getting scheduler timesteps for generation
        In case of image-to-image generation, it updates number of steps according to strength
        
        Parameters:
           num_inference_steps (int):
              number of inference steps for generation
           strength (float):
               value between 0.0 and 1.0, that controls the amount of noise that is added to the input image. 
               Values that approach 1.0 allow for lots of variations but will also produce images that are not semantically consistent with the input.
        """
        # get the original timestep using init_timestep
   
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start     

    def _encode_prompt(
        self,
        prompt,
        num_images_per_prompt,
        prompt_embeds: None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
        """

        if prompt_embeds is None:

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_embeds = self.text_encoder(text_input_ids, share_inputs=True, share_outputs=True)
            prompt_embeds = torch.from_numpy(prompt_embeds[0])

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        # Don't need to get uncond prompt embedding because of LCM Guided Distillation
        return prompt_embeds

    def run_safety_checker(self, image, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(
                    image, output_type="pil"
                )
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(
                feature_extractor_input, return_tensors="pt"
            )
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concep

    def prepare_latents(
        self,image,timestep,batch_size, num_channels_latents, height, width, dtype, scheduler,latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if image:
            #latents_shape = (1, 4, 512, 512 // 8)
            #input_image, meta = preprocess(image,512,512)
            latents_shape = (1, 4, 512 // 8, 512 // 8)
            noise = np.random.randn(*latents_shape).astype(np.float32)
            input_image,meta = preprocess(image,512,512)
            moments = self.vae_encoder(input_image)[self._vae_e_output]
            mean, logvar = np.split(moments, 2, axis=1)
            std = np.exp(logvar * 0.5)
            latents = (mean + std * np.random.randn(*mean.shape)) * 0.18215
            noise = torch.randn(shape, dtype=dtype)
            #latents = scheduler.add_noise(init_latents, noise, timestep)
            latents = scheduler.add_noise(torch.from_numpy(latents), noise, timestep)
           
        else:
            latents = torch.randn(shape, dtype=dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        return latents

    def get_w_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        see https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298
        Args:
        timesteps: torch.Tensor: generate embedding vectors at these timesteps
        embedding_dim: int: dimension of the embeddings to generate
        dtype: data type of the generated embeddings
        Returns:
        embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        init_image: Optional[PIL.Image.Image] = None,
        strength: Optional[float] = 0.8,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        guidance_scale: float = 7.5,
        scheduler = None,
        num_images_per_prompt: Optional[int] = 1,
        latents: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 4,
        lcm_origin_steps: int = 50,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        model: Optional[Dict[str, any]] = None,
        seed: Optional[int] = 1234567,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback = None,
        callback_userdata = None
    ):

        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if seed is not None:
            torch.manual_seed(seed)

        #print("After Step 1: batch size is ", batch_size)
        # do_classifier_free_guidance = guidance_scale > 0.0
        # In LCM Implementation:  cfg_noise = noise_cond + cfg_scale * (noise_cond - noise_uncond) , (cfg_scale > 0.0 using CFG)

        # 2. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            prompt_embeds=prompt_embeds,
        )
        #print("After Step 2: prompt embeds is ", prompt_embeds)
        #print("After Step 2: scheduler is ", scheduler )
        # 3. Prepare timesteps
        #scheduler.set_timesteps(num_inference_steps, original_inference_steps=lcm_origin_steps)
        latent_timestep = None
        if init_image:
            scheduler.set_timesteps(num_inference_steps, original_inference_steps=lcm_origin_steps)
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, scheduler)
            latent_timestep = timesteps[:1]
        else:
             scheduler.set_timesteps(num_inference_steps, original_inference_steps=lcm_origin_steps)
             timesteps = scheduler.timesteps
        #timesteps = scheduler.timesteps
        #latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        #print("timesteps: ", latent_timestep)

        #print("After Step 3: timesteps is ", timesteps)

        # 4. Prepare latent variable
        num_channels_latents = 4
        latents = self.prepare_latents(
                init_image,
                latent_timestep,
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                scheduler,
                latents,
            )
        
        latents = latents * scheduler.init_noise_sigma

        #print("After Step 4: ")
        bs = batch_size * num_images_per_prompt

        # 5. Get Guidance Scale Embedding
        w = torch.tensor(guidance_scale).repeat(bs)
        w_embedding = self.get_w_embedding(w, embedding_dim=256)
        #print("After Step 5: ")
        # 6. LCM MultiStep Sampling Loop:
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if callback:
                    callback(i+1, callback_userdata)

                ts = torch.full((bs,), t, dtype=torch.long)

                # model prediction (v-prediction, eps, x)
                model_pred = self.unet([latents, ts, prompt_embeds, w_embedding],share_inputs=True, share_outputs=True)[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents, denoised = scheduler.step(
                    torch.from_numpy(model_pred), t, latents, return_dict=False
                )
                progress_bar.update()

        #print("After Step 6: ")

        vae_start = time.time()

        if not output_type == "latent":
            image = torch.from_numpy(self.vae_decoder(denoised / 0.18215, share_inputs=True, share_outputs=True)[0])
        else:
            image = denoised

        print("Decoder Ended: ", time.time() - vae_start)
        #post_start = time.time()

        #if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
        #else:
        #    do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        #print ("After do_denormalize: image is ", image)

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        return image[0]
    
class StableDiffusionEngineReferenceOnly(DiffusionPipeline):
    def __init__(
            self,
            #scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
            model="bes-dev/stable-diffusion-v1-4-openvino",
            tokenizer="openai/clip-vit-large-patch14",
            device=["CPU","CPU","CPU"]
            ):
        #self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer)
        try: 
            self.tokenizer = CLIPTokenizer.from_pretrained(model,local_files_only=True)
        except:
            self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer)
            self.tokenizer.save_pretrained(model)
                
        #self.scheduler = scheduler
        # models
     
        self.core = Core()
        self.core.set_property({'CACHE_DIR': os.path.join(model, 'cache')}) #adding caching to reduce init time
        # text features

        print("Text Device:",device[0])
        self.text_encoder = self.core.compile_model(os.path.join(model, "text_encoder.xml"), device[0])
        
        self._text_encoder_output = self.text_encoder.output(0)
       
        # diffusion
        print("unet_w Device:",device[1])
        self.unet_w = self.core.compile_model(os.path.join(model, "unet_reference_write.xml"), device[1]) 
        self._unet_w_output = self.unet_w.output(0)
        self.latent_shape = tuple(self.unet_w.inputs[0].shape)[1:]
        
        print("unet_r Device:",device[1])
        self.unet_r = self.core.compile_model(os.path.join(model, "unet_reference_read.xml"), device[1]) 
        self._unet_r_output = self.unet_r.output(0)
        # decoder
        print("Vae Device:",device[2])
        
        self.vae_decoder = self.core.compile_model(os.path.join(model, "vae_decoder.xml"), device[2])
            
        # encoder
            
        self.vae_encoder = self.core.compile_model(os.path.join(model, "vae_encoder.xml"), device[2]) 
    
        self.init_image_shape = tuple(self.vae_encoder.inputs[0].shape)[2:]

        self._vae_d_output = self.vae_decoder.output(0)
        self._vae_e_output = self.vae_encoder.output(0) if self.vae_encoder is not None else None  

        self.height = self.unet_w.input(0).shape[2] * 8
        self.width = self.unet_w.input(0).shape[3] * 8      



    def __call__(
            self,
            prompt,
            image = None,
            negative_prompt=None,
            scheduler=None,
            strength = 1.0,
            num_inference_steps = 32,
            guidance_scale = 7.5,
            eta = 0.0,
            create_gif = False,
            model = None,
            callback = None,
            callback_userdata = None
    ):
        # extract condition
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        text_embeddings = self.text_encoder(text_input.input_ids)[self._text_encoder_output]
    

        # do classifier free guidance
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
        
            if negative_prompt is None:
                uncond_tokens = [""]
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = negative_prompt
                
            tokens_uncond = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length, #truncation=True,  
                return_tensors="np"
            )
            uncond_embeddings = self.text_encoder(tokens_uncond.input_ids)[self._text_encoder_output]
            text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, scheduler)
        latent_timestep = timesteps[:1]

        ref_image = self.prepare_image(
            image=image,
            width=512,
            height=512,
        )
        # get the initial random noise unless the user supplied it
        latents, meta = self.prepare_latents(None, latent_timestep, scheduler)
        #ref_image_latents, _ = self.prepare_latents(init_image, latent_timestep, scheduler)
        ref_image_latents = self.ov_prepare_ref_latents(ref_image)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        if create_gif:
            frames = []        

        for i, t in enumerate(self.progress_bar(timesteps)):
            if callback:
               callback(i, callback_userdata)

            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            
            # ref only part
            noise = randn_tensor(
                ref_image_latents.shape
            )
               
            ref_xt = scheduler.add_noise(
                torch.from_numpy(ref_image_latents),
                noise,
                t.reshape(
                    1,
                ),    
            ).numpy()
            ref_xt = np.concatenate([ref_xt] * 2) if do_classifier_free_guidance else ref_xt
            ref_xt = scheduler.scale_model_input(ref_xt, t)

            # MODE = "write"
            result_w_dict = self.unet_w([
                ref_xt,
                t,
                text_embeddings
            ])
            down_0_attn0 = result_w_dict["/unet/down_blocks.0/attentions.0/transformer_blocks.0/norm1/LayerNormalization_output_0"]
            down_0_attn1 = result_w_dict["/unet/down_blocks.0/attentions.1/transformer_blocks.0/norm1/LayerNormalization_output_0"]
            down_1_attn0 = result_w_dict["/unet/down_blocks.1/attentions.0/transformer_blocks.0/norm1/LayerNormalization_output_0"]
            down_1_attn1 = result_w_dict["/unet/down_blocks.1/attentions.1/transformer_blocks.0/norm1/LayerNormalization_output_0"]
            down_2_attn0 = result_w_dict["/unet/down_blocks.2/attentions.0/transformer_blocks.0/norm1/LayerNormalization_output_0"]
            down_2_attn1 = result_w_dict["/unet/down_blocks.2/attentions.1/transformer_blocks.0/norm1/LayerNormalization_output_0"]
            mid_attn0    = result_w_dict["/unet/mid_block/attentions.0/transformer_blocks.0/norm1/LayerNormalization_output_0"]
            up_1_attn0   = result_w_dict["/unet/up_blocks.1/attentions.0/transformer_blocks.0/norm1/LayerNormalization_output_0"]
            up_1_attn1   = result_w_dict["/unet/up_blocks.1/attentions.1/transformer_blocks.0/norm1/LayerNormalization_output_0"]
            up_1_attn2   = result_w_dict["/unet/up_blocks.1/attentions.2/transformer_blocks.0/norm1/LayerNormalization_output_0"]
            up_2_attn0   = result_w_dict["/unet/up_blocks.2/attentions.0/transformer_blocks.0/norm1/LayerNormalization_output_0"]
            up_2_attn1   = result_w_dict["/unet/up_blocks.2/attentions.1/transformer_blocks.0/norm1/LayerNormalization_output_0"]
            up_2_attn2   = result_w_dict["/unet/up_blocks.2/attentions.2/transformer_blocks.0/norm1/LayerNormalization_output_0"]
            up_3_attn0   = result_w_dict["/unet/up_blocks.3/attentions.0/transformer_blocks.0/norm1/LayerNormalization_output_0"]
            up_3_attn1   = result_w_dict["/unet/up_blocks.3/attentions.1/transformer_blocks.0/norm1/LayerNormalization_output_0"]
            up_3_attn2   = result_w_dict["/unet/up_blocks.3/attentions.2/transformer_blocks.0/norm1/LayerNormalization_output_0"]
                
            # MODE = "read"
            noise_pred = self.unet_r([
                latent_model_input, t, text_embeddings, down_0_attn0, down_0_attn1, down_1_attn0,
                down_1_attn1, down_2_attn0, down_2_attn1, mid_attn0, up_1_attn0, up_1_attn1, up_1_attn2, 
                up_2_attn0, up_2_attn1, up_2_attn2, up_3_attn0, up_3_attn1, up_3_attn2
            ])[0]
                
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred[0], noise_pred[1]
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs)["prev_sample"].numpy()
     
            if create_gif:
                frames.append(latents)
              
        if callback:
            callback(num_inference_steps, callback_userdata)

        # scale and decode the image latents with vae
        
        image = self.vae_decoder(latents)[self._vae_d_output]
      
        image = self.postprocess_image(image, meta)

        if create_gif:
            gif_folder=os.path.join(model,"../../../gif")
            if not os.path.exists(gif_folder):
                os.makedirs(gif_folder)
            for i in range(0,len(frames)):
                image = self.vae_decoder(frames[i])[self._vae_d_output]
                image = self.postprocess_image(image, meta)
                output = gif_folder + "/" + str(i).zfill(3) +".png"
                cv2.imwrite(output, image)
            with open(os.path.join(gif_folder, "prompt.json"), "w") as file:
                json.dump({"prompt": prompt}, file)
            frames_image =  [Image.open(image) for image in glob.glob(f"{gif_folder}/*.png")]  
            frame_one = frames_image[0]
            gif_file=os.path.join(gif_folder,"stable_diffusion.gif")
            frame_one.save(gif_file, format="GIF", append_images=frames_image, save_all=True, duration=100, loop=0)

        return image
    
    def ov_prepare_ref_latents(self, refimage, vae_scaling_factor=0.18215):
        #refimage = refimage.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        moments = self.vae_encoder(refimage)[0]
        mean, logvar = np.split(moments, 2, axis=1)
        std = np.exp(logvar * 0.5)
        ref_image_latents = (mean + std * np.random.randn(*mean.shape))
        ref_image_latents = vae_scaling_factor * ref_image_latents
        #ref_image_latents = scheduler.add_noise(torch.from_numpy(ref_image_latents), torch.from_numpy(noise), latent_timestep).numpy()
        
        # aligning device to prevent device errors when concating it with the latent model input
        #ref_image_latents = ref_image_latents.to(device=device, dtype=dtype)
        return ref_image_latents
    
    def prepare_latents(self, image:PIL.Image.Image = None, latent_timestep:torch.Tensor = None, scheduler = LMSDiscreteScheduler):
        """
        Function for getting initial latents for starting generation
        
        Parameters:
            image (PIL.Image.Image, *optional*, None):
                Input image for generation, if not provided randon noise will be used as starting point
            latent_timestep (torch.Tensor, *optional*, None):
                Predicted by scheduler initial step for image generation, required for latent image mixing with nosie
        Returns:
            latents (np.ndarray):
                Image encoded in latent space
        """
        latents_shape = (1, 4, self.height // 8, self.width // 8)
   
        noise = np.random.randn(*latents_shape).astype(np.float32)
        if image is None:
            #print("Image is NONE")
            # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
            if isinstance(scheduler, LMSDiscreteScheduler):
             
                noise = noise * scheduler.sigmas[0].numpy()
                return noise, {}
            elif isinstance(scheduler, EulerDiscreteScheduler):
              
                noise = noise * scheduler.sigmas.max().numpy()
                return noise, {}
            else:
                return noise, {}
        input_image, meta = preprocess(image,self.height,self.width)
       
        moments = self.vae_encoder(input_image)[self._vae_e_output]
      
        mean, logvar = np.split(moments, 2, axis=1)
  
        std = np.exp(logvar * 0.5)
        latents = (mean + std * np.random.randn(*mean.shape)) * 0.18215
       
         
        latents = scheduler.add_noise(torch.from_numpy(latents), torch.from_numpy(noise), latent_timestep).numpy()
        return latents, meta

    def postprocess_image(self, image:np.ndarray, meta:Dict):
        """
        Postprocessing for decoded image. Takes generated image decoded by VAE decoder, unpad it to initila image size (if required), 
        normalize and convert to [0, 255] pixels range. Optionally, convertes it from np.ndarray to PIL.Image format
        
        Parameters:
            image (np.ndarray):
                Generated image
            meta (Dict):
                Metadata obtained on latents preparing step, can be empty
            output_type (str, *optional*, pil):
                Output format for result, can be pil or numpy
        Returns:
            image (List of np.ndarray or PIL.Image.Image):
                Postprocessed images

                        if "src_height" in meta:
            orig_height, orig_width = meta["src_height"], meta["src_width"]
            image = [cv2.resize(img, (orig_width, orig_height))
                        for img in image]
    
        return image
        """
        if "padding" in meta:
            pad = meta["padding"]
            (_, end_h), (_, end_w) = pad[1:3]
            h, w = image.shape[2:]
            #print("image shape",image.shape[2:])
            unpad_h = h - end_h
            unpad_w = w - end_w
            image = image[:, :, :unpad_h, :unpad_w]
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = (image[0].transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)

           

        if "src_height" in meta:
            orig_height, orig_width = meta["src_height"], meta["src_width"]
            image = cv2.resize(image, (orig_width, orig_height))
                        
        return image

        
                      #image = (image / 2 + 0.5).clip(0, 1)
        #image = (image[0].transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)   


    def get_timesteps(self, num_inference_steps:int, strength:float, scheduler):
        """
        Helper function for getting scheduler timesteps for generation
        In case of image-to-image generation, it updates number of steps according to strength
        
        Parameters:
           num_inference_steps (int):
              number of inference steps for generation
           strength (float):
               value between 0.0 and 1.0, that controls the amount of noise that is added to the input image. 
               Values that approach 1.0 allow for lots of variations but will also produce images that are not semantically consistent with the input.
        """
        # get the original timestep using init_timestep
   
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start 
    def prepare_image(
        self,
        image,
        width,
        height,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        if not isinstance(image, np.ndarray):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            if isinstance(image[0], PIL.Image.Image):
                images = []

                for image_ in image:
                    image_ = image_.convert("RGB")
                    image_ = image_.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)

                image = images

                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = (image - 0.5) / 0.5
                image = image.transpose(0, 3, 1, 2)
            elif isinstance(image[0], np.ndarray):
                image = np.concatenate(image, dim=0)

        if do_classifier_free_guidance and not guess_mode:
            image = np.concatenate([image] * 2)

        return image

def print_npu_turbo_art():
    random_number = random.randint(1, 3)
    
    if random_number == 1:
        print("                                                                                                                      ")
        print("      ___           ___         ___                                ___           ___                         ___      ")
        print("     /\  \         /\  \       /\  \                              /\  \         /\  \         _____         /\  \     ")
        print("     \:\  \       /::\  \      \:\  \                ___          \:\  \       /::\  \       /::\  \       /::\  \    ")
        print("      \:\  \     /:/\:\__\      \:\  \              /\__\          \:\  \     /:/\:\__\     /:/\:\  \     /:/\:\  \   ")
        print("  _____\:\  \   /:/ /:/  /  ___  \:\  \            /:/  /      ___  \:\  \   /:/ /:/  /    /:/ /::\__\   /:/  \:\  \  ")
        print(" /::::::::\__\ /:/_/:/  /  /\  \  \:\__\          /:/__/      /\  \  \:\__\ /:/_/:/__/___ /:/_/:/\:|__| /:/__/ \:\__\ ")
        print(" \:\~~\~~\/__/ \:\/:/  /   \:\  \ /:/  /         /::\  \      \:\  \ /:/  / \:\/:::::/  / \:\/:/ /:/  / \:\  \ /:/  / ")
        print("  \:\  \        \::/__/     \:\  /:/  /         /:/\:\  \      \:\  /:/  /   \::/~~/~~~~   \::/_/:/  /   \:\  /:/  /  ")
        print("   \:\  \        \:\  \      \:\/:/  /          \/__\:\  \      \:\/:/  /     \:\~~\        \:\/:/  /     \:\/:/  /   ")
        print("    \:\__\        \:\__\      \::/  /                \:\__\      \::/  /       \:\__\        \::/  /       \::/  /    ")
        print("     \/__/         \/__/       \/__/                  \/__/       \/__/         \/__/         \/__/         \/__/     ")
        print("                                                                                                                      ")
    elif random_number == 2:
        print(" _   _   ____    _   _     _____   _   _   ____    ____     ___  ")
        print("| \ | | |  _ \  | | | |   |_   _| | | | | |  _ \  | __ )   / _ \ ")
        print("|  \| | | |_) | | | | |     | |   | | | | | |_) | |  _ \  | | | |")
        print("| |\  | |  __/  | |_| |     | |   | |_| | |  _ <  | |_) | | |_| |")
        print("|_| \_| |_|      \___/      |_|    \___/  |_| \_\ |____/   \___/ ")
        print("                                                                 ")
    else:
        print("")
        print("    )   (                                 (                )   ")
        print(" ( /(   )\ )              *   )           )\ )     (    ( /(   ")
        print(" )\()) (()/(      (     ` )  /(      (   (()/(   ( )\   )\())  ")
        print("((_)\   /(_))     )\     ( )(_))     )\   /(_))  )((_) ((_)\   ")
        print(" _((_) (_))    _ ((_)   (_(_())   _ ((_) (_))   ((_)_    ((_)  ")
        print("| \| | | _ \  | | | |   |_   _|  | | | | | _ \   | _ )  / _ \  ")
        print("| .` | |  _/  | |_| |     | |    | |_| | |   /   | _ \ | (_) | ")
        print("|_|\_| |_|     \___/      |_|     \___/  |_|_\   |___/  \___/  ")
        print("                                                               ")



