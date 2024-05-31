# Copyright 2023 The HuggingFace Team.
# Adapted by Lei Li and Angela Dai.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# https://github.com/huggingface/diffusers/blob/v0.21.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py

import os
import inspect
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import cv2
import torch
import torch.nn.functional as F
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    LoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import (
    AsymmetricAutoencoderKL,
    AutoencoderKL,
    UNet2DConditionModel,
)
from diffusers.models.attention_processor import Attention
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers import DPMSolverMultistepScheduler
from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)

logger = logging.get_logger(__name__)


class AttentionStore:

    def __init__(self, attn_res):
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.curr_step_index = 0
        self.attn_res = attn_res

    def reset(self):
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    @staticmethod
    def get_empty_store():
        return {"down": [], "mid": [], "up": []}

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= 0 and is_cross:
            if attn.shape[1] == np.prod(self.attn_res):
                self.step_store[place_in_unet].append(attn)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()

    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def aggregate_attention(self, from_where: List[str]) -> torch.Tensor:
        out = []
        attention_maps = self.get_average_attention()
        for location in from_where:
            for item in attention_maps[location]:
                cross_maps = item.reshape(
                    -1, self.attn_res[0], self.attn_res[1], item.shape[-1]
                )
                out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out


class AttnProcessor2_0:

    def __init__(self, attnstore, place_in_unet, do_cfg):
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet
        self.do_cfg = do_cfg
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            asmask = attention_mask
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )
        else:
            asmask = None

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        if self.attnstore is not None:
            attn_scores = attn.get_attention_scores(
                query=query.contiguous().view(batch_size * attn.heads, -1, head_dim),
                key=key.contiguous().view(batch_size * attn.heads, -1, head_dim),
                attention_mask=asmask,
            )
            if self.do_cfg and batch_size == 2:
                src_len, tgt_len = attn_scores.shape[-2:]
                attn_scores = attn_scores.view(batch_size, attn.heads, src_len, tgt_len)
                attn_scores_uncond, attn_scores_text = attn_scores.chunk(2)
                attn_scores = attn_scores_text.view(
                    batch_size // 2 * attn.heads, src_len, tgt_len
                )
            else:
                assert batch_size == 1
            self.attnstore(attn_scores, is_cross, self.place_in_unet)

        return hidden_states


class StableDiffusionInpaintPipeline(
    DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, FromSingleFileMixin
):
    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor"]
    _exclude_from_cpu_offload = ["safety_checker"]

    def __init__(
        self,
        vae: Union[AutoencoderKL, AsymmetricAutoencoderKL],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if (
            hasattr(scheduler.config, "steps_offset")
            and scheduler.config.steps_offset != 1
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate(
                "steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if (
            hasattr(scheduler.config, "skip_prk_steps")
            and scheduler.config.skip_prk_steps is False
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration"
                " `skip_prk_steps`. `skip_prk_steps` should be set to True in the configuration file. Please make"
                " sure to update the config accordingly as not setting `skip_prk_steps` in the config might lead to"
                " incorrect results in future versions. If you have downloaded this checkpoint from the Hugging Face"
                " Hub, it would be very nice if you could open a Pull request for the"
                " `scheduler/scheduler_config.json` file"
            )
            deprecate(
                "skip_prk_steps not set",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            new_config = dict(scheduler.config)
            new_config["skip_prk_steps"] = True
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(
            unet.config, "_diffusers_version"
        ) and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse(
            "0.9.0.dev0"
        )
        is_unet_sample_size_less_64 = (
            hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        )
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely .If you're checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate(
                "sample_size<64", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        if unet.config.in_channels != 9:
            logger.info(
                f"You have loaded a UNet with {unet.config.in_channels} input channels which."
            )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

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

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=prompt_embeds_dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

        return prompt_embeds, negative_prompt_embeds

    def run_safety_checker(self, image, device, dtype):
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
            ).to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        strength,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if strength < 0 or strength > 1:
            raise ValueError(
                f"The value of strength should in [0.0, 1.0] but is {strength}"
            )

        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if (callback_steps is None) or (
            callback_steps is not None
            and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
        is_strength_max=True,
        return_noise=False,
        return_image_latents=False,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        if return_image_latents or (latents is None and not is_strength_max):
            image = image.to(device=device, dtype=dtype)

            if image.shape[1] == 4:
                image_latents = image
            else:
                image_latents = self._encode_vae_image(image=image, generator=generator)

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = (
                noise
                if is_strength_max
                else self.scheduler.add_noise(image_latents, noise, timestep)
            )
            latents = (
                latents * self.scheduler.init_noise_sigma
                if is_strength_max
                else latents
            )
        else:
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma

        outputs = (latents,)

        if return_noise:
            outputs += (noise,)

        if return_image_latents:
            outputs += (image_latents,)

        return outputs

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                self.vae.encode(image[i : i + 1]).latent_dist.sample(
                    generator=generator[i]
                )
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = self.vae.encode(image).latent_dist.sample(
                generator=generator
            )

        image_latents = self.vae.config.scaling_factor * image_latents

        return image_latents

    def prepare_mask_latents(
        self,
        mask,
        masked_image,
        batch_size,
        height,
        width,
        dtype,
        device,
        generator,
        do_classifier_free_guidance,
    ):
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)

        if masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            masked_image_latents = self._encode_vae_image(
                masked_image, generator=generator
            )

        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(
                batch_size // masked_image_latents.shape[0], 1, 1, 1
            )

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2)
            if do_classifier_free_guidance
            else masked_image_latents
        )

        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents

    def _update_mask_latents(
        self,
        image,
        attention,
        attn_thresh,
        batch_size,
        num_images_per_prompt,
        generator,
        do_classifier_free_guidance,
        pool_size=1,
    ):
        assert image.ndim == 4 and attention.ndim == 2
        image = image.to(attention)
        height, width = image.shape[-2:]
        dtype = attention.dtype
        device = attention.device

        mask = attention.view((1, 1) + attention.shape)
        mask = F.avg_pool2d(
            mask,
            kernel_size=(pool_size * 2 + 1, pool_size * 2 + 1),
            stride=(1, 1),
            padding=(pool_size, pool_size),
        )
        mask = F.interpolate(mask, size=(height, width), mode="bicubic")

        attn_thresh = torch.quantile(mask.float(), attn_thresh, interpolation="linear")
        mask = (mask >= attn_thresh).to(device=device, dtype=dtype)

        masked_image = image * (mask < 0.5)

        mask, masked_image_latents = self.prepare_mask_latents(
            mask,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )
        return mask, masked_image_latents

    def register_attention_control(self, do_cfg):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue
            cross_att_count += 1
            attn_procs[name] = AttnProcessor2_0(
                attnstore=self.attention_store,
                place_in_unet=place_in_unet,
                do_cfg=do_cfg,
            )
        self.unet.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = cross_att_count

    def get_indices(self, prompt: str) -> Dict[str, int]:
        ids = self.tokenizer(prompt).input_ids
        indices = {
            i: tok
            for tok, i in zip(
                self.tokenizer.convert_ids_to_tokens(ids), range(len(ids))
            )
        }
        return indices

    def get_timesteps(self, num_inference_steps, strength, device):
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        token_indices: List[int] = None,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        masked_image_latents: torch.FloatTensor = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 1.0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = False,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        attn_res: Optional[Tuple[int]] = (16, 16),
        attn_threshs: List[float] = None,
        attn_average_steps: int = 1,
        dynamic_mask_start: int = 0,
        dynamic_mask_stop: int = 20,
    ):

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt,
            height,
            width,
            strength,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None)
            if cross_attention_kwargs is not None
            else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps, strength=strength, device=device
        )
        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        is_strength_max = strength == 1.0

        init_image = self.image_processor.preprocess(image, height=height, width=width)
        init_image = init_image.to(dtype=torch.float32)

        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.unet.config.in_channels
        return_image_latents = num_channels_unet == 4

        latents_outputs = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            image=init_image,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            return_noise=True,
            return_image_latents=return_image_latents,
        )

        if return_image_latents:
            latents, noise, image_latents = latents_outputs
        else:
            latents, noise = latents_outputs

        mask_condition = self.mask_processor.preprocess(
            mask_image, height=height, width=width
        )

        if masked_image_latents is None:
            masked_image = init_image * (mask_condition < 0.5)
        else:
            masked_image = masked_image_latents

        mask, masked_image_latents = self.prepare_mask_latents(
            mask_condition,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )

        if num_channels_unet == 9:
            num_channels_mask = mask.shape[1]
            num_channels_masked_image = masked_image_latents.shape[1]
            if (
                num_channels_latents + num_channels_mask + num_channels_masked_image
                != self.unet.config.in_channels
            ):
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                    f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                    f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                    " `pipeline.unet` or your `mask_image` or `image` input."
                )
        elif num_channels_unet != 4:
            raise ValueError(
                f"The unet {self.unet.__class__} should have either 4 or 9 input channels, not {self.unet.config.in_channels}."
            )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if attn_res is None:
            attn_res = int(np.ceil(width / 32)), int(np.ceil(height / 32))
        self.attention_store = AttentionStore(attn_res)
        self.register_attention_control(do_cfg=do_classifier_free_guidance)
        if dynamic_mask_stop > dynamic_mask_start:
            assert len(attn_threshs) == dynamic_mask_stop - dynamic_mask_start

        all_attns = list()
        all_masks = list()

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )

                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                if num_channels_unet == 9:
                    latent_model_input = torch.cat(
                        [latent_model_input, mask, masked_image_latents], dim=1
                    )

                all_masks.append(mask[0, 0])

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                attn = self.attention_store.aggregate_attention(
                    from_where=("up", "down", "mid")
                )
                all_attns.append(attn[:, :, token_indices])

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                if num_channels_unet == 4:
                    init_latents_proper = image_latents[:1]
                    init_mask = mask[:1]

                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents_proper = self.scheduler.add_noise(
                            init_latents_proper, noise, torch.tensor([noise_timestep])
                        )

                    latents = (
                        1 - init_mask
                    ) * init_latents_proper + init_mask * latents

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

                if dynamic_mask_start <= i < dynamic_mask_stop:
                    assert attn_average_steps > 0
                    average_start = max(i - (attn_average_steps - 1), 0)
                    mask, masked_image_latents = self._update_mask_latents(
                        image=init_image,
                        attention=torch.stack(all_attns[average_start:], dim=0)
                        .mean(dim=0)
                        .sum(dim=-1),
                        attn_thresh=attn_threshs[i - dynamic_mask_start],
                        batch_size=batch_size,
                        num_images_per_prompt=num_images_per_prompt,
                        generator=generator,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                    )

        all_attns = torch.stack(all_attns, dim=0)
        if output_type == "pil" or output_type == "np":
            all_attns = all_attns.detach().cpu().float().numpy()

        all_masks = torch.stack(all_masks, dim=0)
        if output_type == "pil" or output_type == "np":
            all_masks = all_masks.detach().cpu().float().numpy()

        if not output_type == "latent":
            condition_kwargs = {}
            if isinstance(self.vae, AsymmetricAutoencoderKL):
                init_image = init_image.to(
                    device=device, dtype=masked_image_latents.dtype
                )
                init_image_condition = init_image.clone()
                init_image = self._encode_vae_image(init_image, generator=generator)
                mask_condition = mask_condition.to(
                    device=device, dtype=masked_image_latents.dtype
                )
                condition_kwargs = {
                    "image": init_image_condition,
                    "mask": mask_condition,
                }
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor,
                return_dict=False,
                **condition_kwargs,
            )[0]
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype
            )
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )
        latent = latents
        if output_type == "pil" or output_type == "np":
            latent = latent.detach().cpu().float().numpy()

        self.maybe_free_model_hooks()

        assert not return_dict
        return (image, latent, all_masks, all_attns)


def get_ldm_inpaint(ldm_inpaint_path, device):
    ldm_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
        ldm_inpaint_path,
        requires_safety_checker=False,
        torch_dtype=torch.float16,
    )
    ldm_inpaint.scheduler = DPMSolverMultistepScheduler.from_config(
        ldm_inpaint.scheduler.config
    )
    ldm_inpaint.safety_checker = None
    ldm_inpaint.set_progress_bar_config(disable=True)
    ldm_inpaint.to(device)
    return ldm_inpaint


def dilate_masks(masks, kernel_size, iterations):
    dtype = masks.dtype
    is_batch = masks.ndim == 3

    if not is_batch:
        masks = np.expand_dims(masks, axis=0)
    assert masks.ndim == 3

    res = list()
    for idx in range(len(masks)):
        mask = (masks[idx] * 255).astype(np.uint8)
        mask = cv2.dilate(
            mask, kernel=np.ones(kernel_size, np.uint8), iterations=iterations
        )
        res.append(mask)
    res = np.stack(res, axis=0)
    res = res > 0

    if not is_batch:
        res = np.squeeze(res, axis=0)

    return res.astype(dtype)


def inpainting(
    ldm_inpaint_path,
    ldm_inpaint,
    image_paths,
    mask_paths,
    out_dir,
    num_inpaints_per_image,
    prompt,
    negative_prompt,
    token_indices,
    num_inference_steps,
    guidance_scale,
    attn_res,
    attn_threshs,
    attn_average_steps,
    dynamic_mask_start,
    dynamic_mask_stop,
    dilate_size,
    dilate_iterations,
    gpu_id,
    seed,
    deterministic,
    callback=None,
    callback_steps=1,
):
    bench_flag = torch.backends.cudnn.benchmark
    deter_flag = torch.are_deterministic_algorithms_enabled()
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    device = f"cuda:{gpu_id}"
    randg = torch.Generator(device)
    randg.manual_seed(seed)

    if ldm_inpaint is None:
        ldm_inpaint = get_ldm_inpaint(ldm_inpaint_path, device)

    for sidx in range(len(image_paths)):
        image_path = image_paths[sidx]
        image_name = Path(image_path).name
        image = np.array(PIL.Image.open(image_path).convert("RGB"))
        assert image.ndim == 3
        assert image.shape[0] == image.shape[1]
        image_size = image.shape[0]
        image = torch.from_numpy(image)
        image = image.unsqueeze(0).permute(0, 3, 1, 2).float() / 255

        if mask_paths[sidx] is not None:
            mask = np.array(PIL.Image.open(mask_paths[sidx]).convert("L"))
            assert mask.ndim == 2
            assert mask.shape[0] == mask.shape[1] == image_size
            mask = mask.astype(np.float32) / 255.0
            if dilate_iterations > 0:
                mask = dilate_masks(
                    masks=mask, kernel_size=dilate_size, iterations=dilate_iterations
                )
            mask = torch.from_numpy(mask).unsqueeze(0)
        else:
            mask = torch.zeros((image_size, image_size))
            mask = mask.unsqueeze(0).float()

        num_tokens = len(ldm_inpaint.get_indices(prompt))
        token_indices = [num_tokens + ti if ti < 0 else ti for ti in token_indices]

        for inpaint_id in range(num_inpaints_per_image):
            inpaint_images, inpaint_latents, inpaint_masks, attn_maps = ldm_inpaint(
                prompt=prompt,
                token_indices=token_indices,
                image=image,
                mask_image=mask,
                height=image_size,
                width=image_size,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                generator=randg,
                output_type="np",
                return_dict=False,
                attn_res=attn_res,
                attn_threshs=attn_threshs,
                attn_average_steps=attn_average_steps,
                dynamic_mask_start=dynamic_mask_start,
                dynamic_mask_stop=dynamic_mask_stop,
                callback=callback,
                callback_steps=callback_steps,
            )
            inpaint_image = np.squeeze(inpaint_images, 0)
            inpaint_image = PIL.Image.fromarray((inpaint_image * 255).astype(np.uint8))
            inpaint_path = os.path.join(
                out_dir.format(f"inpaint{inpaint_id:03d}"), image_name
            )
            inpaint_image.save(inpaint_path)

            mask_folder = inpaint_path[:-4]
            os.makedirs(mask_folder, mode=0o777)
            for midx in range(0, inpaint_masks.shape[0], 5):
                inpaint_mask = np.expand_dims(inpaint_masks[midx], 2)
                inpaint_mask = np.tile(inpaint_mask, (1, 1, 3))
                inpaint_mask = PIL.Image.fromarray(
                    (inpaint_mask * 255).astype(np.uint8)
                )
                inpaint_mask = inpaint_mask.resize(
                    (image_size, image_size), resample=PIL.Image.Resampling.NEAREST
                )
                inpaint_mask.save(os.path.join(mask_folder, f"mask_{midx:03d}.png"))
    torch.backends.cudnn.benchmark = bench_flag
    torch.use_deterministic_algorithms(deter_flag)
