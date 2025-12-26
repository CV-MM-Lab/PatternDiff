from networkx import is_valid_degree_sequence_havel_hakimi
import torch
import sys

import sys
sys.path.append('./')

from diffusers import  DDIMScheduler, AutoencoderKL,ControlNetModel,StableDiffusionControlNetInpaintPipeline
from PIL import Image
from transformers import CLIPTokenizer,CLIPVisionModelWithProjection


import os
from tqdm import tqdm
import torch
import numpy as np
import torchvision.transforms as T
from diffusers.pipelines import DiffusionPipeline
from torch.utils.data import DataLoader
from src.utils.image_composition import compose_img_cpu
import os
from adapter.adapter import PatternDiff
from src.datasetsPatternDiff.tatal import FashionPatternDataset
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a demo script.")
    parser.add_argument(
        "--input_ckpt",
        type=str,
        default='./weight/PatternDiff.bin',
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='../stable-diffusion-inpainting',
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default='./image_encoder',
    )
    parser.add_argument(
        "--datasets_root",
        type=str,
        default='./data/',

    )
    parser.add_argument(
        "--controlnet_path",
        type=str,
        default='../control_v11p_sd15_openpose',
    )
    parser.add_argument(
        "--save_result_path",
        type=str,
        default='./vitonhd/unpaired', 
    )
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    save_result_path = args.save_result_path
    os.makedirs(save_result_path,exist_ok=True)
    pretrained_model_name_or_path = args.pretrained_model_name_or_path
    image_encoder_path = args.image_encoder_path
    ip_ckpt = args.input_ckpt
  
    datasets_root = args.datasets_root
    controlnet_path = args.controlnet_path
    device = "cuda"
    val_scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    val_scheduler.set_timesteps(50, device=device)
    tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer",torch_dtype=torch.float16
        )
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae",torch_dtype=torch.float16)
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(device,torch.float16)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        pretrained_model_name_or_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        scheduler=val_scheduler,
        vae=vae,
        safety_checker=None
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
                dataroot_path=datasets_root,
                phase='test',
                order='unpaired',
                radius=5,
                sketch_threshold_range=(20, 20),
                tokenizer=tokenizer,
                dataset_c = ['vitonhd'],
                size=(512, 384)
            )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=1,
        collate_fn=my_collate,
    )
    model = PatternDiff(pipe, ip_ckpt, device,image_encoder,m=3)
   

    for batch in tqdm(train_dataloader):
        model_img = batch["image"]
        mask_img = batch["inpaint_mask"]
        mask_img = mask_img.type(torch.float32)
        prompts = batch["original_captions"]  
        image = batch["texture"]
        openpose = batch['openpose']
        ext = ".jpg"
        pattern = batch['pattern']
        pattern_source = batch['pattern_source']

        images = model.generate_inpainting(image=model_img,pil_image=image, pattern=pattern,mask_image=mask_img,prompt=prompts,num_samples=1, num_inference_steps=50, seed=seed,openpose=openpose,pattern_source = pattern_source)
        for i in range(len(images)):
            final_img = images[i]
            model_i = model_img[i] * 0.5 + 0.5
            final_img = compose_img_cpu(model_i, images[i], batch['parse_array'][i])
            final_img = T.functional.to_pil_image(final_img)
            final_img.save(
                os.path.join(save_result_path, batch["im_name"][i].replace(".jpg", ext)))
            
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