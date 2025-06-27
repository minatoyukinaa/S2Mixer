import os
import sys

import imageio
import numpy as np
import torch
from model.S2Mixer import S2Mixer_2x,S2Mixer
import torch.nn.functional as F
def process_img(pic1):
    for i in range(pic1.shape[2]):
        a, b = np.percentile(pic1[:, :, i], (1, 99))
        pic1[:, :, i] = np.clip(pic1[:, :, i], a, b)
        pic1[:, :, i] = (pic1[:, :, i] - a) / (b - a)
    pic1 = (pic1 * 255).astype(np.uint8)
    return pic1
def save_png_60m(band60, save_path):
    # 第一个参数,gdal.GetDriverByName('GTiff'), 第二个是数组格式是band60m
    bands = np.zeros((band60.shape[0], band60.shape[1], 3))


    bands[:, :, 0] = band60[:, :, 1]
    bands[:, :, 1] = band60[:, :, 1]
    bands[:, :, 2] = band60[:, :, 0]
    bands = bands.astype(np.uint8)
    imageio.imwrite(save_path, bands)
    pass
def save_png_20m(band20, save_path):
    bands = np.zeros((band20.shape[0], band20.shape[1], 3))


    bands[:, :, 0] = band20[:, :, 1]
    bands[:, :, 1] = band20[:, :, 2]
    bands[:, :, 2] = band20[:, :, 0]
    bands = bands.astype(np.uint8)
    imageio.imwrite(save_path, bands)
    pass
S2Mixer_2x = S2Mixer_2x()
S2Mixer_6x = S2Mixer()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 1. loading checkpoint
ckpt_2x = torch.load("ckpt/S2Mixer_2x/ckpt.pth")
S2Mixer_2x.load_state_dict(ckpt_2x)
ckpt_6x = torch.load("ckpt/S2Mixer_6x/ckpt.pth")
S2Mixer_6x.load_state_dict(ckpt_6x)
S2Mixer_2x,S2Mixer_6x = S2Mixer_2x.to(device),S2Mixer_6x.to(device)

output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# 6x results
file = 'data/6x/119.pt'
data10, data20, data60 = torch.load(file, weights_only=True)[0:3]
d10, d20, d60 = data10.clone().unsqueeze(0), data20.clone().unsqueeze(0), data60.clone().unsqueeze(0)
d10,d20,d60 = d10.to(device), d20.to(device),d60.to(device)
output = S2Mixer_6x(d10, d20, d60).detach().cpu().squeeze().permute(1, 2, 0).numpy()
output = process_img(output)
save_png_60m(output,output_dir+'/S2Mixer_6x.png')

# 2x results
file = 'data/2x/35.pt'
data10, data20, data60 = torch.load(file, weights_only=True)[0:3]
d10, d20, d60 = data10.clone().unsqueeze(0), data20.clone().unsqueeze(0), data60.clone().unsqueeze(0)
d10,d20,d60 = d10.to(device), d20.to(device),d60.to(device)
output = S2Mixer_2x(d10, d20).detach().cpu().squeeze().permute(1, 2, 0).numpy()
output = process_img(output)
print(f'shape of output:{output.shape}')
save_png_20m(output,output_dir+'/S2Mixer_2x.png')

# 6x bicubic
file = 'data/6x/119.pt'
data10, data20, data60 = torch.load(file, weights_only=True)[0:3]
d60 = data60.clone().unsqueeze(0)
output = F.interpolate(d60,scale_factor=6,mode="bicubic").cpu().squeeze().permute(1, 2, 0).numpy()
output = process_img(output)
save_png_60m(output,output_dir+'/bicubic_6x.png')

# 2x bicubic
file = 'data/2x/35.pt'
data10, data20, data60 = torch.load(file, weights_only=True)[0:3]
d20 = data20.clone().unsqueeze(0)
output = F.interpolate(d20,scale_factor=2,mode="bicubic").cpu().squeeze().permute(1, 2, 0).numpy()
output = process_img(output)
save_png_20m(output,output_dir+'/bicubic_2x.png')

