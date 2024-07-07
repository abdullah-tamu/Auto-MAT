# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""
import copy
import cv2
import pyspng
import glob
import os
import random
from typing import List, Optional
import click
from tqdm import tqdm
import dnnlib
import numpy as np
import PIL.Image
import torch
import legacy
from networks.mat import Generator


def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = {name: tensor for name, tensor in named_params_and_buffers(src_module)}
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)


def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())


def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())


def read_image(image_path):
    with open(image_path, 'rb') as f:
        if pyspng is not None and image_path.endswith('.png'):
            image = pyspng.load(f.read())
        else:
            image = np.array(PIL.Image.open(f))
        dim = (512, 512)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]  # HW => HWC
        image = np.repeat(image, 3, axis=2)
    image = image.transpose(2, 0, 1)  # HWC => CHW
    image = image[:3]
    return image


def generate_mask(pr, pix_w, show_mask=False):
    mask_w = 512
    pix_w = int(2 ** np.floor(np.log2(pix_w)))
    n_divs = int(np.floor(mask_w / pix_w))
    rnd_mask_small = np.random.binomial(1, pr, (n_divs, n_divs))

    rnd_mask = np.zeros((mask_w, mask_w))

    indexes = np.where(rnd_mask_small == 0)

    for i in range(0, n_divs):
        for j in range(0, n_divs):
            rnd_mask[i * pix_w:i * pix_w + pix_w, j * pix_w:j * pix_w + pix_w] = rnd_mask_small[i, j]

    if (show_mask):
        cv2.imshow('s', rnd_mask * 255)
        cv2.waitKey()

    return rnd_mask


def refine_mask(dif_img, mask, th, pix_w):
    dim = mask.shape
    mask_w = 512
    pix_w = int(2 ** np.floor(np.log2(pix_w)))
    n_divs = int(np.floor(mask_w / pix_w))
    dif_img_small = np.zeros((n_divs, n_divs))
    if (pix_w == 1):
        dif_img[dif_img < th] = 0
        dif_img[dif_img >= th] = 10
    else:
        for i in range(0, n_divs):
            for j in range(0, n_divs):
                dif_img_small[i, j] = np.sum(dif_img[i * pix_w:i * pix_w + pix_w, j * pix_w:j * pix_w + pix_w])
                sum_val = np.sum(dif_img[i * pix_w:i * pix_w + pix_w, j * pix_w:j * pix_w + pix_w])
                if (sum_val < th):
                    dif_img[i * pix_w:i * pix_w + pix_w, j * pix_w:j * pix_w + pix_w] = 0
                else:
                    dif_img[i * pix_w:i * pix_w + pix_w, j * pix_w:j * pix_w + pix_w] = 10

            # # rnd_mask[i*pix_w:i*pix_w+pix_w,j*pix_w:j*pix_w+pix_w]=dif_img_small[i,j]
    mask[0][0][dif_img == 0] = 1

    return mask


device = 'cuda:0'

def inpaint_imge(ipath, G,
                 mpath_cleft: Optional[str],
                 truncation_psi: float,
                 noise_mode: str,
                 outdir: str):
    label = torch.zeros([1, G.c_dim], device=device)
    iname = os.path.basename(ipath)
    if (iname != 'ee.png' and "Style" not in iname):
        print(f'Prcessing: {iname}')

        mask = cv2.imread(mpath_cleft, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        dim = (512, 512)

        mother_mask = torch.ones(dim).float().to(device).unsqueeze(0).unsqueeze(0)
        pr = 0.7
        pix_w = 1
        th = 2 * pix_w * pix_w

        z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)

        difference_sum = np.zeros(dim)

        image = read_image(ipath)
        image = mask * (mask / 255) + image * (1 - (mask / 255))

        image_01 = torch.from_numpy(image).permute(1, 2, 0).cpu().numpy()
        image = (torch.from_numpy(image).float().to(device) / 127.5 - 1).unsqueeze(0)

        mask = cv2.imread(mpath_cleft, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        dim = (512, 512)
        mask = cv2.resize(mask, dim, interpolation=cv2.INTER_AREA)
        y = 0
        for k in tqdm(range(0, 60)):
            rnd_mask = generate_mask(pr, pix_w)
            rnd_mask[mask == 1] = 1

            mother_mask[0][0][rnd_mask == 0.0] = 0.0
            x = 0
            while (x < 1):
                output = G(image, mother_mask, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                # replace the image with its inpainted version
                output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
                output = output[0].cpu().numpy()

                dif = np.asarray(np.abs(np.mean(image_01, axis=2) - np.mean(output, axis=2)), dtype=np.uint8)

                mother_mask = refine_mask(dif, mother_mask, th, pix_w)

                x = x + 1
                m_mask = np.asarray((1 - mother_mask.cpu().numpy()) * 255, dtype=np.uint8)[0][0]
                ######### write the mask to disk ##############
                # cv2.imwrite(f'masks/{iname}_{y}_{x}.png', m_mask)

            difference_sum = difference_sum + dif

            y = y + 1

        ws = G.mapping(z, label)
        mask = mother_mask.clone()

        torch.cuda.empty_cache()

        blur = cv2.GaussianBlur(difference_sum, (1, 1), 0)

        face_pixels = difference_sum.copy()
        face_pixels[mask[0][0].cpu().numpy() == 1] = 0
        npix = 512 * 512 - np.sum(mask[0][0].cpu().numpy())
        reduction_factor = 0.9
        idx_optimal = reduction_factor * (np.sum(face_pixels) / npix)

        m = blur.copy()

        m[m < idx_optimal] = -1
        m[m >= idx_optimal] = 0
        m[m == -1] = 1

        mask = torch.from_numpy(m).float().to(device).unsqueeze(0).unsqueeze(0)

        generated_images = G.synthesis(image, mask, ws, noise_mode='const')
        generated_images = (generated_images.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(
            torch.uint8)
        generated_images = generated_images[0].cpu().numpy()
        PIL.Image.fromarray(generated_images, 'RGB').save(f'{outdir}/{iname}')
        return generated_images
    return 0


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--dpath', help='the path of the input image', required=True)
@click.option('--mpath', help='the path of the mask')
@click.option('--resolution', type=int, help='resolution of input image', default=512, show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const',
              show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str,default='./output', required=True, metavar='DIR')
def normalize_AutoMAT(
        ctx: click.Context,
        network_pkl: str,
        dpath: str,
        mpath: Optional[str],
        resolution: int,
        truncation_psi: float,
        noise_mode: str,
        outdir: str,
):
    seed = 250  # pick up a random number
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print(f'Loading data from: {dpath}')
    img_list = sorted(glob.glob(dpath + '/*.png') + glob.glob(dpath + '/*.jpg') + glob.glob(dpath + '/*.jpeg'))

    print(f'Loading networks from: {network_pkl}')
    device = torch.device('cuda')
    # device = torch.device('cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G_saved = legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False)  # type: ignore
    net_res = 512 if resolution > 512 else resolution
    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=net_res, img_channels=3).to(
        device).eval().requires_grad_(False)
    copy_params_and_buffers(G_saved, G, require_all=True)
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float()  # type: ignore

    os.makedirs(outdir, exist_ok=True)

    for i, ipath in enumerate(img_list):
        res = inpaint_imge(ipath, G, mpath, truncation_psi, noise_mode, outdir)


if __name__ == "__main__":
    normalize_AutoMAT()
# ----------------------------------------------------------------------------
