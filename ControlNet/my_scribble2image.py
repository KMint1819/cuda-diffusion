from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
from pathlib import Path
from argparse import ArgumentParser
import copy

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('/home/control_sd15_scribble.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = np.zeros_like(img, dtype=np.uint8)
        detected_map[np.min(img, axis=2) < 127] = 255

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results

def main(args):
    input_image = cv2.cvtColor(cv2.imread(str(args.img_path)), cv2.COLOR_BGR2RGB) 
    prompt = args.prompt
    a_prompt = 'best quality, extremely detailed'
    n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
    num_samples = 1
    image_resolution = 512
    ddim_steps = 20
    guess_mode = False
    strength = 1.0
    scale = 9.0
    seed = -1
    eta = 0.0

    ips = [
        input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta]
    print(f'ips: {ips[0].shape}, {ips[1:]}')
    outs = process(*ips)
    for i, out in enumerate(outs):
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(args.out_path / f'output-{i}.png'), out)
    print("Images saved to ", str(args.out_path.absolute()))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--img_path', type=str, default='../media/turtle.png', help='path to input image')
    parser.add_argument('-o', '--out_path', type=str, default='/build', help='path to output directory')
    parser.add_argument('-p', '--prompt', type=str, default='turtle')
    args = parser.parse_args()

    args.img_path = Path(args.img_path)
    args.out_path = Path(args.out_path)
    main(args)

