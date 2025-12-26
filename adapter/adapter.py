import os
from turtle import width
from typing import List
import sys


import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from adapter.utils import is_torch2_available, get_generator
import torch.nn.functional as F
import math
 


from adapter.attention_processor import HAM, AttnProcessor,IPAttnProcessor



class ImageProjModelSPP(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4,m=3):
        super().__init__()
        self.m = m
        self.num_levels = 4
        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = int(clip_extra_context_tokens / 2)
        self.proj = torch.nn.Linear(clip_embeddings_dim, int(self.clip_extra_context_tokens +self.m)  * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        #self.norm_spp = torch.nn.LayerNorm(120)
        #self.spp = SPPLayer()
        self.spp_linear = torch.nn.Linear(120,int(self.clip_extra_context_tokens - self.m)  * cross_attention_dim)

    def forward(self, image_embeds,spps):
        with torch.no_grad():
            cur_spp = []
            for spp in spps:
                spp = F.interpolate(spp, scale_factor=4, mode='nearest')
                s = self.spp_fn(spp)
                cur_spp.append(s)
            cur_spp = torch.stack(cur_spp,dim=0)
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens + self.m, self.cross_attention_dim
        )
 
        #cur_spp = self.norm_spp(cur_spp)
        cpp_feature = self.spp_linear(cur_spp).reshape(
            -1, self.clip_extra_context_tokens - self.m, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(torch.cat([clip_extra_context_tokens,cpp_feature],dim=1))
        #clip_extra_context_tokens = self.norm(clip_extra_context_tokens+cpp_feature)
        return clip_extra_context_tokens
    
    def spp_fn(self,x):
        num, c, h, w = x.size() 
        for i in range(self.num_levels):
            level = i+1
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            pooling = (math.floor((kernel_size[0]*level-h+1)/2), math.floor((kernel_size[1]*level-w+1)/2))
 
        
            tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
 
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten




class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )
        
    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class PatternDiff:
    def __init__(self,sd_pipe, ip_ckpt, device, image_encoder,m=3,num_tokens=8):
        self.device = device
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        self.image_encoder =image_encoder

        self.pipe = sd_pipe.to(self.device)
        self.vae = self.pipe.vae
        self.m = m
        self.set_ip_adapter()

        self.image_proj_model = self.init_proj()

        self.pattern_proj_model = self.init_proj_pattern()

        self.load_ip_adapter()

    def init_proj(self):
        model = ImageProjModel(cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
                          clip_extra_context_tokens=self.num_tokens).to(self.device, dtype=torch.float16) 
        return model
    
    def init_proj_pattern(self):
        model = ImageProjModelSPP(cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
                          clip_extra_context_tokens=8,m=self.m).to(self.device, dtype=torch.float16) 
        return model

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:

                attn_procs[name] = HAM(hidden_size=hidden_size,
                                                      cross_attention_dim=cross_attention_dim,
                                                      scale=1.0,
                                                      num_tokens=self.num_tokens,).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe,"controlnet"):
            controlnet = self.pipe.controlnet
            attn_procs_controlnet = {}
            for name in controlnet.attn_processors.keys():
                cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = unet.config.block_out_channels[block_id]
                if cross_attention_dim is None:
                    attn_procs_controlnet[name] = AttnProcessor()
                else:
                    attn_procs_controlnet[name] = HAM(hidden_size=hidden_size,
                                                      cross_attention_dim=cross_attention_dim,
                                                      scale=1.0,
                                                      num_tokens=self.num_tokens,).to(self.device, dtype=torch.float16)
            controlnet.set_attn_processor(attn_procs_controlnet)


    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        self.pattern_proj_model.load_state_dict(state_dict['ip_pattern'])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_sd"])
        ip_control_layers = torch.nn.ModuleList(self.pipe.controlnet.attn_processors.values())
        ip_control_layers.load_state_dict(state_dict["ip_controlnet"])


    def get_image_embeds(self,pil_image=None):
        clip_image_embeds = pil_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image_embeds)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds.image_embeds)
        uncond_image_prompt_embeds = torch.randn_like(image_prompt_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds
    def get_pattern_embeds(self,pil_image=None,latents_spp=None):
        clip_image_embeds = pil_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image_embeds).image_embeds
        image_prompt_embeds = self.pattern_proj_model(clip_image_embeds,latents_spp)
        uncond_image_prompt_embeds = torch.randn_like(image_prompt_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds


    def generate_inpainting(
            self,
            image=None,
            pil_image=None,
            pattern=None,
            clip_image_embeds=None,
            prompt=None,
            negative_prompt=None,
            num_samples=4,
            seed=None,
            guidance_scale=7.5,
            num_inference_steps=30,
            openpose=None,
            mask_image=None,
            pattern_source=None,
            **kwargs,
    ):
        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        
        cpp_p = []    
        for p in pattern_source:
            latents_spp1 = self.vae.encode(p.unsqueeze(0).to(self.device,dtype=torch.float16)).latent_dist.sample()
            cpp_p.append(latents_spp1)

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image
        )
        pattern_embed,uncond_pattern_embed = self.get_pattern_embeds(pil_image=pattern,latents_spp=cpp_p)

        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds,pattern_embed], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds,uncond_pattern_embed], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            image=image,
            mask_image=mask_image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            control_image=openpose,
            **kwargs,
        ).images

        return images