from email.policy import strict
import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
import sys
sys.path.append('./')
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel,ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
import sys
import numpy as np


from adapter.adapter import ImageProjModel,ImageProjModelSPP
from src.datasetsPatternDiff.tatal import FashionPatternDataset

from diffusers.optimization import get_scheduler

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import prepare_mask_and_masked_image



from adapter.attention_processor import HAM,AttnProcessor


class Adapter(torch.nn.Module):

    def __init__(self, unet, controlnet, image_texture_model,image_pattern_model, controlnet_adapter_modules ,adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_texture_model = image_texture_model
        self.image_pattern_model = image_pattern_model
        self.adapter_modules = adapter_modules
        self.controlnet_adapter_modules = controlnet_adapter_modules
        self.controlnet = controlnet

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds,image_embeds_pattern,controlnet_image,spp_latent):
        ip_tokens = self.image_texture_model(image_embeds)
        ip_tokens_pattern = self.image_pattern_model(image_embeds_pattern,spp_latent)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens,ip_tokens_pattern], dim=1)
        # controlnet
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents[:,:4,:,:],
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_image,
            return_dict=False,
        )
        # unet predict the noise residual
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=[
                sample.to(dtype=noisy_latents.dtype) for sample in down_block_res_samples
            ],
            mid_block_additional_residual=mid_block_res_sample.to(dtype=noisy_latents.dtype),
            return_dict=False,
        )[0]
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    # parser.add_argument(
    #     "--data_json_file",
    #     type=str,
    #     default=None,
    #     required=True,
    #     help="Training data",
    #)
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--controlnet_model_path",
        type=str,
        default=None,
        required=True,
        help="Path to controlnet"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=800)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)
    

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer",torch_dtype=torch.float16)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder",torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae",torch_dtype=torch.float16)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    x = torch.load('../stable-diffusion-inpainting/unet/diffusion_pytorch_model.fp16.bin')
    x = torch.load(os.path.join(args.pretrained_model_name_or_path,'unet','diffusion_pytorch_model.fp16.bin'))
    config = UNet2DConditionModel.load_config(args.pretrained_model_name_or_path, subfolder="unet")
    unet = UNet2DConditionModel.from_config(config)
    unet.load_state_dict(x,strict=True)
  
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_path, torch_dtype=torch.float16)


    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.requires_grad_(False)
    image_encoder.requires_grad_(False)



    image_proj_model = ImageProjModel(cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=8)
    image_pattern_model = ImageProjModelSPP( cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=8,m=3)


    # unet atten_procs
    attn_procs = {}
    unet_sd = unet.state_dict()
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
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                "to_k_ip_p.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip_p.weight": unet_sd[layer_name + ".to_v.weight"],

            }
            attn_procs[name] = HAM(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,num_tokens=8)
            attn_procs[name].load_state_dict(weights,strict=False)

    attn_procs_controlnet = {}
    unet_controlnet = controlnet.state_dict()
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
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_controlnet[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_controlnet[layer_name + ".to_v.weight"],
                "to_k_ip_p.weight": unet_controlnet[layer_name + ".to_k.weight"],
                "to_v_ip_p.weight": unet_controlnet[layer_name + ".to_v.weight"],

            }
            attn_procs_controlnet[name] = HAM(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,scale = 1.0,num_tokens=8)
            attn_procs_controlnet[name].load_state_dict(weights,strict=False)





    unet.set_attn_processor(attn_procs)
    controlnet.set_attn_processor(attn_procs_controlnet)

    unet_adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    controlnet_adapter_modules = torch.nn.ModuleList(controlnet.attn_processors.values())

    adapter = Adapter(unet=unet,
                           controlnet=controlnet,
                           image_texture_model=image_proj_model,
                           image_pattern_model=image_pattern_model, 
                           controlnet_adapter_modules=controlnet_adapter_modules,
                           adapter_modules=unet_adapter_modules,
                           ckpt_path=args.pretrained_ip_adapter_path)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    # optimizer
    params_to_opt = itertools.chain(adapter.image_texture_model.parameters(), 
                                    adapter.image_pattern_model.parameters(),
                                    adapter.controlnet_adapter_modules.parameters(),
                                    adapter.adapter_modules.parameters())
    
    
  
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=150 * accelerator.num_processes,
        num_training_steps=150000 * accelerator.num_processes,
    )


    def my_collate(batch):
        keys = batch[0].keys()
        total = {}
        for key in keys:
            total[key] = []
        for item in batch:
            cur_keys = item.keys()
            for cur_key in cur_keys:
                if cur_key == 'parse_array':
                    total[cur_key].append(torch.from_numpy(item[cur_key]))
                else:   
                    total[cur_key].append(item[cur_key])
        for key in keys:
            if key =='c_name' or key =='im_name' or key =='im_name_change' or key =='original_captions' or key =='pattern_source' or key == 'im_parse' or key == 'category':
                continue
            total[key] = torch.stack(total[key],dim=0)
        return total
    train_dataset = FashionPatternDataset(
            dataroot_path=args.data_root_path,
            phase='train',
            order='paired',
            radius=5,
            sketch_threshold_range=(20, 20),
            tokenizer=tokenizer,
            size=(512, 384)
        )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=my_collate
    )

    # Prepare everything with our `accelerator`.
    adapter, optimizer, train_dataloader,lr_scheduler= accelerator.prepare(adapter, optimizer, train_dataloader,lr_scheduler)

    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(adapter):
                # Convert images to latent space
                with torch.no_grad():
                    init_image = batch["image"]
                    spp_latent = []
                    for spp in batch['pattern_source']:
                        latents_spp = vae.encode(spp.unsqueeze(0).to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                        spp_latent.append(latents_spp)
                    latents = vae.encode(
                        init_image.to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    mask_img = batch["inpaint_mask"]
                    mask = mask_img.type(weight_dtype)
                    masked_image = init_image * (mask < 0.5)
                    _, _, height, width = init_image.shape
                    mask = torch.nn.functional.interpolate(
                        mask, size=(height // 8, width // 8)
                    )
                    mask = mask.to(accelerator.device, dtype=weight_dtype)
                    masked_image_latents = vae.encode(
                        masked_image.to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    masked_image_latents = masked_image_latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                with torch.no_grad():
                    controlnet_image = batch['openpose'].to(accelerator.device,dtype=weight_dtype)
                    encoder_hidden_states = text_encoder(batch["captions"].to(accelerator.device))[0]
                    
                    image_embeds = batch['texture'].to(accelerator.device,dtype=weight_dtype)
                    image_embeds_pattern = batch['pattern'].to(accelerator.device,dtype=weight_dtype)
                    image_texture = image_encoder(image_embeds).image_embeds
                    image_pattern = image_encoder(image_embeds_pattern).image_embeds
                if unet.config.in_channels == 9:
                    noisy_latents = torch.cat([noisy_latents, mask, masked_image_latents], dim=1)

                noise_pred = adapter(noisy_latents, timesteps, encoder_hidden_states, image_texture,image_pattern,controlnet_image,spp_latent)
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()

                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    print("Epoch {}, step {}, global_step {}, data_time: {}, time: {}, step_loss: {}".format(
                        epoch, step, global_step,load_data_time, time.perf_counter() - begin, avg_loss))

            global_step += 1

            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path,safe_serialization=False)

            begin = time.perf_counter()


if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    main()


'''
accelerate launch --multi_gpu --mixed_precision "fp16" --main_process_port 29502 ./train.py --pretrained_model_name_or_path="../stable-diffusion-inpainting"  --image_encoder_path="../PatternDiff/image_encoder" --resolution=512  --train_batch_size=3  --dataloader_num_workers=10 --learning_rate=1e-04  --weight_decay=0.01  --output_dir="./weight"  --save_steps=10 --controlnet_model_path="../control_v11p_sd15_openpose" --data_root_path='/home/ys/Desktop/ys/patternfashion/' 

'''