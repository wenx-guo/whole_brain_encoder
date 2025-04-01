import os
import argparse
import numpy as np
from PIL import Image
import scipy as sp
import pandas as pd
import xarray as xr
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import (
    alexnet,
    AlexNet_Weights,
    inception_v3,
    Inception_V3_Weights,
    efficientnet_b1,
    EfficientNet_B1_Weights,
)
from torchvision.models.feature_extraction import create_feature_extractor
import clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

str2bool = lambda x: x.lower() == "true"
parser = argparse.ArgumentParser()
parser.add_argument("--subj", type=int, default=1)
parser.add_argument("--retrieval_res_dir", type=str, default="/engram/nklab/wg2361/nsd")
parser.add_argument("--avg", type=str2bool, default=True)
parser.add_argument("--method", type=str, default="cos")
input_args = parser.parse_args()
subj = input_args.subj
avg = input_args.avg
retrieval_res_dir = input_args.retrieval_res_dir
method = input_args.method

nsd_root = (
    "/engram/nklab/datasets/natural_scene_dataset/model_training_datasets/neural_data/"
)
model_pred_root = "/engram/nklab/algonauts/ethan/whole_brain_encoder/results/schaefer/enc_1_3_5_7_run_1_2"
imagenet_pred_root = os.path.join(
    model_pred_root,
    "imagenet",
    f"subj_{subj:02}",
)
avg_str = "avg" if avg else "single_trials"
retrieval_res_dir = os.path.join(retrieval_res_dir, f"subj_{subj:02}", avg_str)

metadata = np.load(
    os.path.join(nsd_root, f"metadata_sub-{subj:02}.npy"), allow_pickle=True
).item()

# load NSD test images and their betas
nsd_images = np.load(os.path.join(retrieval_res_dir, "nsd_imgs.npy"))
nsd_images = torch.from_numpy(nsd_images / 255.0).permute(0, 3, 1, 2).to(device)
IMSIZE = nsd_images.shape[-1]
nsd_betas = np.load(os.path.join(retrieval_res_dir, "cat_betas.npy"))

# load predicted betas and imagenet-retrieved images
pred_path = os.path.join(model_pred_root, f"subj_{subj:02}", "test.npy")
d = np.load(
    pred_path, allow_pickle=True
).item()  # image paths are wrong in this npy so we don't check
predicted_betas = []
for hemi in ["lh", "rh"]:
    la = np.zeros(163842, dtype=bool)
    for roi in metadata[f"{hemi}_rois"]:
        la = np.logical_or(la, metadata[f"{hemi}_rois"][roi])
    predicted_betas.append(d["out"][hemi][:, la])
predicted_betas = np.concatenate(predicted_betas, axis=1)
assert predicted_betas.shape[0] == nsd_betas.shape[0]

retrieved_paths = xr.load_dataarray(
    os.path.join(retrieval_res_dir, f"imagenet_top10_{method}_paths.nc")
)
retrieved_paths = retrieved_paths.sel(top_k=0)
predicted_images = []
for path in retrieved_paths.values:
    im = torch.from_numpy(np.array(Image.open(path)) / 255.0)
    if im.ndim == 2:  # convert grayscale to RGB
        im = im.unsqueeze(0).repeat(3, 1, 1)
    elif im.ndim == 3:
        im = im.permute(2, 0, 1)
    im = transforms.Resize(
        (IMSIZE, IMSIZE), interpolation=transforms.InterpolationMode.BILINEAR
    )(im)
    assert im.shape == (3, IMSIZE, IMSIZE)
    predicted_images.append(im)
predicted_images = torch.stack(predicted_images).to(device)

assert predicted_images.shape == nsd_images.shape


@torch.no_grad()
def two_way_identification(
    all_brain_recons, all_images, model, preprocess, feature_layer=None, return_avg=True
):
    preds = model(
        torch.stack([preprocess(recon) for recon in all_brain_recons], dim=0).to(device)
    )
    reals = model(
        torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device)
    )
    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()

    r = np.corrcoef(reals, preds)
    r = r[: len(all_images), len(all_images) :]
    congruents = np.diag(r)

    success = r < congruents
    success_cnt = np.sum(success, 0)

    if return_avg:
        perf = np.mean(success_cnt) / (len(all_images) - 1)
        return perf
    else:
        return success_cnt, len(all_images) - 1


res = []
# 1. pixel correlation
print("calculating pixel correlation...")
nsd_images_flattened = nsd_images.reshape(len(nsd_images), -1).cpu()
nsd_images_centered = nsd_images_flattened - nsd_images_flattened.mean(
    dim=1, keepdim=True
)
predicted_images_flattened = predicted_images.reshape(len(predicted_images), -1).cpu()
predicted_images_centered = (
    predicted_images_flattened - predicted_images_flattened.mean(dim=1, keepdim=True)
)
pixcorr = torch.matmul(nsd_images_centered, predicted_images_centered.T)
pixcorr /= (
    nsd_images_centered.norm(dim=1, keepdim=True)
    * predicted_images_centered.norm(dim=1).unsqueeze(0)
    + 1e-8
)
pixcorr = pixcorr.diag().cpu().numpy()
res.append(pd.Series(pixcorr, index=retrieved_paths.image_id, name="PixCorr"))

# 2. SSIM score
# convert image to grayscale with rgb2grey
print("calculating SSIM...")
nsd_images_gray = rgb2gray(nsd_images.permute((0, 2, 3, 1)).cpu())
predicted_images_gray = rgb2gray(predicted_images.permute((0, 2, 3, 1)).cpu())

ssim_score = []
for nsd_img, pred_im in zip(nsd_images_gray, predicted_images_gray):
    ssim_score.append(
        ssim(
            pred_im,
            nsd_img,
            multichannel=True,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            data_range=1.0,
        )
    )
res.append(pd.Series(ssim_score, index=retrieved_paths.image_id, name="SSIM"))

# 3. AlexNet feature correlation
print("calculating AlexNet feature correlation...")
alex_weights = AlexNet_Weights.IMAGENET1K_V1
alex_model = create_feature_extractor(
    alexnet(weights=alex_weights), return_nodes=["features.4", "features.11"]
).to(device)
alex_model.eval().requires_grad_(False)

preprocess = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

alexnet2 = two_way_identification(
    predicted_images.float(), nsd_images.float(), alex_model, preprocess, "features.4"
)
res.append(pd.Series(alexnet2, index=retrieved_paths.image_id, name="Alex(2)"))

alexnet5 = two_way_identification(
    predicted_images.float(), nsd_images.float(), alex_model, preprocess, "features.11"
)
res.append(pd.Series(alexnet5, index=retrieved_paths.image_id, name="Alex(5)"))
del alex_model, alex_weights

# 4. Inception
print("calculating Inception feature correlation...")
inception_weights = Inception_V3_Weights.DEFAULT
inception_model = create_feature_extractor(
    inception_v3(weights=inception_weights), return_nodes=["avgpool"]
).to(device)
inception_model.eval().requires_grad_(False)
preprocess = transforms.Compose(
    [
        transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

inception = two_way_identification(
    predicted_images.float(), nsd_images.float(), inception_model, preprocess, "avgpool"
)
res.append(pd.Series(inception, index=retrieved_paths.image_id, name="Incep"))
del inception_model, inception_weights

# 5. CLIP
print("calculating CLIP feature correlation...")
clip_model, preprocess = clip.load("ViT-L/14", device=device)
preprocess = transforms.Compose(
    [
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ]
)
clip_res = two_way_identification(
    predicted_images.float(),
    nsd_images.float(),
    clip_model.encode_image,
    preprocess,
    None,
)  # final layer
res.append(pd.Series(clip_res, index=retrieved_paths.image_id, name="CLIP"))
del clip_model

# 6. EfficientNet
print("calculating EfficientNet feature correlation...")
eff_weights = EfficientNet_B1_Weights.DEFAULT
eff_model = create_feature_extractor(
    efficientnet_b1(weights=eff_weights), return_nodes=["avgpool"]
).to(device)
eff_model.eval().requires_grad_(False)

preprocess = transforms.Compose(
    [
        transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
gt = eff_model(preprocess(nsd_images.float()))["avgpool"]
gt = gt.reshape(len(gt), -1).cpu().numpy()
fake = eff_model(preprocess(predicted_images.float()))["avgpool"]
fake = fake.reshape(len(fake), -1).cpu().numpy()

effnet = np.array(
    [sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(len(gt))]
)
res.append(pd.Series(effnet, index=retrieved_paths.image_id, name="Eff"))

# 7. SwAV model
print("calculating SwAV feature correlation...")
swav_model = torch.hub.load("facebookresearch/swav:main", "resnet50")
swav_model = create_feature_extractor(swav_model, return_nodes=["avgpool"]).to(device)
swav_model.eval().requires_grad_(False)

preprocess = transforms.Compose(
    [
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

gt = swav_model(preprocess(nsd_images.float()))["avgpool"]
gt = gt.reshape(len(gt), -1).cpu().numpy()
fake = swav_model(preprocess(predicted_images.float()))["avgpool"]
fake = fake.reshape(len(fake), -1).cpu().numpy()

swav = np.array(
    [sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(len(gt))]
)
res.append(pd.Series(swav, index=retrieved_paths.image_id, name="SwAV"))

res = pd.concat(res, axis=1)
res.to_csv(os.path.join(retrieval_res_dir, f"imagenet_retrieval_{method}_eval.csv"))
