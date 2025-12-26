import torch
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a weight process script.")
    parser.add_argument(
        "--input_model_dir",
        type=str,
        default='./weight/checkpoint',
    )
    parser.add_argument(
        "--save_model_dir",
        type=str,
        default='./weight',
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    ckpt = args.input_model_dir
    save_path = args.save_model_dir
    path = os.path.join(save_path,'model.bin')
    sd = torch.load(os.path.join(ckpt,'pytorch_model.bin'), map_location="cpu")
    image_proj_sd = {}
    image_pattern_sd = {}
    ip_sd = {}
    ip_controlnet = {}
    up_blocks = {}
    for k in sd:
        if k.startswith("unet"):
            continue
        elif k.startswith("image_texture_model"):
            image_proj_sd[k.replace("image_texture_model.", "")] = sd[k]
        elif k.startswith("image_pattern_model"):
            image_pattern_sd[k.replace("image_pattern_model.","")] = sd[k]
        elif k.startswith("controlnet_adapter_modules"):
            ip_controlnet[k.replace("controlnet_adapter_modules.","")] = sd[k]
        elif k.startswith("adapter_modules"):
            ip_sd[k.replace("adapter_modules.","")] = sd[k]
        
    torch.save({"image_proj": image_proj_sd,"ip_pattern":image_pattern_sd,"ip_sd": ip_sd, "ip_controlnet": ip_controlnet,"up_blocks":up_blocks}, path)


if __name__ == "__main__":
    main()