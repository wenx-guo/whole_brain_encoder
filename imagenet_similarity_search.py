import os
from pathlib import Path
import argparse
import numpy as np
import xarray as xr
from tqdm import tqdm
import pandas as pd
import glob
import torch
import torch.nn.functional as F
from collections import defaultdict

from utils.args import get_default_args
from datasets.nsd import nsd_dataset_avg, nsd_dataset

str2bool = lambda x: x.lower() == "true"
parser = argparse.ArgumentParser()
parser.add_argument("--subj", type=int, default=1)
parser.add_argument("--save_dir", type=str, default="/engram/nklab/wg2361/nsd")
parser.add_argument("--avg", type=str2bool, default=True)
parser.add_argument("--top_k", type=int, default=10)
input_args = parser.parse_args()
subj = input_args.subj
save_dir = input_args.save_dir
avg = input_args.avg
top_k = input_args.top_k

nsd_root = (
    "/engram/nklab/datasets/natural_scene_dataset/model_training_datasets/neural_data/"
)
imagenet_pred_root = os.path.join(
    "/engram/nklab/algonauts/ethan/whole_brain_encoder/results/schaefer/enc_1_3_5_7_run_1_2/imagenet",
    f"subj_{subj:02}",
)
save_str = "avg" if avg else "single_trials"
save_dir = os.path.join(save_dir, f"subj_{subj:02}", save_str)
Path(save_dir).mkdir(parents=True, exist_ok=True)

metadata = np.load(
    os.path.join(nsd_root, f"metadata_sub-{subj:02}.npy"), allow_pickle=True
).item()

# load the NSD test set
args = get_default_args()
args.subj = subj
imgs = []
betas = defaultdict(list)
for i_hemi, hemi in enumerate(["lh", "rh"]):
    args.hemi = hemi

    # load the ROI mask
    la = np.zeros(163842, dtype=bool)
    for roi in metadata[f"{hemi}_rois"]:
        la = np.logical_or(la, metadata[f"{hemi}_rois"][roi])

    test_dataset = nsd_dataset_avg(args, transform=None, split="test")
    test_dataset.backbone_arch = False
    if i_hemi == 0:
        img_idx = test_dataset.split_presented_imgs
    else:
        assert np.all(test_dataset.split_presented_imgs == img_idx)

    for i_data, (img, beta) in enumerate(test_dataset):
        if i_hemi == 0:
            imgs.append(img)
        else:
            assert np.all(np.isclose(img, imgs[i_data]))

        betas[hemi].append(beta["betas"][la])

    if i_hemi == 0:
        imgs = np.stack(imgs)
    betas[hemi] = np.stack(betas[hemi])
betas = np.concatenate([betas["lh"], betas["rh"]], axis=1)
n_images = betas.shape[0]
assert betas.shape[0] == img_idx.shape[0] == imgs.shape[0]

betas_torch = torch.from_numpy(betas)
# normalize for cosine similarity
normed_betas = F.normalize(betas_torch, p=2, dim=1)
# Also center for correlation
betas_centered = betas_torch - betas_torch.mean(dim=1, keepdim=True)
del betas_torch

# initialize tracking tensor
size = betas.shape[0]
best_cos_vals = torch.full((size, top_k), -float("inf")).to(torch.float64)
best_cos_paths = np.full((size, top_k), "", dtype=object)

best_corr_vals = torch.full((size, top_k), -float("inf")).to(torch.float64)
best_corr_paths = np.full((size, top_k), "", dtype=object)

imgnet_cls_paths = sorted(glob.glob(os.path.join(imagenet_pred_root, "**.npy")))
for i_cls, cls_path in tqdm(enumerate(imgnet_cls_paths)):
    model_pred = np.load(cls_path, allow_pickle=True).item()
    img_paths = np.array(model_pred["img_paths"])
    activations = np.concatenate(
        [model_pred["out"]["lh"], model_pred["out"]["rh"]], axis=1
    )
    activations = torch.from_numpy(activations)

    # Cosine similarity
    normed_acts = F.normalize(activations, p=2, dim=1)  # (N_class, 70949)
    cos_sims = torch.matmul(normed_betas, normed_acts.T)  # (515, N_class)

    # Pearson correlation: center the activations
    acts_centered = activations - activations.mean(dim=1, keepdim=True)
    corr_sims = torch.matmul(betas_centered, acts_centered.T)
    corr_sims /= (
        betas_centered.norm(dim=1, keepdim=True)
        * acts_centered.norm(dim=1).unsqueeze(0)
        + 1e-6  # numerical stability
    )

    # Update best cosine similarity
    cls_topk_cos, cls_topk_cos_idxs = torch.topk(cos_sims, k=top_k, dim=1)
    cls_topk_cos_paths = img_paths[cls_topk_cos_idxs.cpu().numpy()]

    combined_cos_vals = torch.cat(
        [best_cos_vals, cls_topk_cos], dim=1
    )  # (n_images, top_k * 2)
    combined_cos_paths = np.concatenate([best_cos_paths, cls_topk_cos_paths], axis=1)

    sorted_indices = torch.argsort(combined_cos_vals, descending=True, dim=1)[
        :, :top_k
    ]  # (n_images, top_k)
    row_idx = torch.arange(n_images).unsqueeze(1)  # (n_images, 1)
    best_cos_vals = combined_cos_vals[
        row_idx, sorted_indices
    ]  # batch indexing when selecting select different columns per row
    best_cos_paths = np.take_along_axis(
        combined_cos_paths, sorted_indices.cpu().numpy(), axis=1
    )

    # Update best correlation
    cls_topk_corr, cls_topk_corr_idxs = torch.topk(corr_sims, k=top_k, dim=1)
    cls_topk_corr_paths = img_paths[cls_topk_corr_idxs.cpu().numpy()]

    combined_corr_vals = torch.cat(
        [best_corr_vals, cls_topk_corr], dim=1
    )  # (n_images, top_k * 2)
    combined_corr_paths = np.concatenate([best_corr_paths, cls_topk_corr_paths], axis=1)

    sorted_indices = torch.argsort(combined_corr_vals, descending=True, dim=1)[
        :, :top_k
    ]  # (n_images, top_k)
    best_corr_vals = combined_corr_vals[
        row_idx, sorted_indices
    ]  # batch indexing when selecting select different columns per row
    best_corr_paths = np.take_along_axis(
        combined_corr_paths, sorted_indices.cpu().numpy(), axis=1
    )

for method in ["cos", "corr"]:
    coords = {
        "image_id": img_idx,
        "top_k": np.arange(top_k),
    }
    val_xr = xr.DataArray(
        best_cos_vals if method == "cos" else best_corr_vals,
        dims=list(coords.keys()),
        coords=coords,
    )
    path_array = best_cos_paths if method == "cos" else best_corr_paths
    path_xr = xr.DataArray(
        np.array(path_array).astype(str),
        dims=list(coords.keys()),
        coords=coords,
    )
    val_xr.to_netcdf(os.path.join(save_dir, f"imagenet_top{top_k}_{method}_vals.nc"))
    path_xr.to_netcdf(os.path.join(save_dir, f"imagenet_top{top_k}_{method}_paths.nc"))

np.save(os.path.join(save_dir, "nsd_imgs.npy"), imgs)
np.save(os.path.join(save_dir, "cat_betas.npy"), betas)
