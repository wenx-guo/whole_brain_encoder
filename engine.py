import math
import sys
from typing import Iterable
import torch
from tqdm import tqdm

from utils.utils import NestedTensor, nested_tensor_from_tensor_list

import utils.utils as utils
import numpy as np


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
):
    model.train()
    print_freq = 1
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", utils.SmoothedValue(window_size=print_freq, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "loss_labels", utils.SmoothedValue(window_size=print_freq)
    )  # , fmt='{value:.2f}'
    header = "Epoch: [{}]".format(epoch)

    for imgs, targets in metric_logger.log_every(data_loader, print_freq, header):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets["lh_betas"][0].to(device, non_blocking=True).to(torch.float32)
        outputs = model(imgs)
        outputs = outputs["pred"]
        loss = criterion(outputs, targets)

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        # loss_value = losses_reduced_scaled.item()
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_value)
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(
            loss=loss_value
        )  # , **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        metric_logger.update(loss_labels=loss_value)  # loss_dict_reduced['loss_recon']
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, data_loader, args, mask, device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "loss_labels", utils.SmoothedValue(window_size=100)
    )  # , fmt='{value:.2f}'
    header = "Test:"

    preds = []
    ys = []

    for imgs, targets in metric_logger.log_every(data_loader, 1, header):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets["lh_betas"][0].to(device, non_blocking=True).to(torch.float32)
        outputs = model(imgs)
        outputs = outputs["pred"]
        loss = criterion(outputs, targets)

        ys.append(targets.flatten(start_dim=1)[:, mask.flatten()])
        preds.append(outputs.flatten(start_dim=1)[:, mask.flatten()])

        loss_value = loss.item()
        metric_logger.update(loss_labels=loss_value)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return torch.vstack(ys), torch.vstack(preds)


@torch.no_grad()
def test(model, criterion, data_loader, args, lh_challenge_rois, rh_challenge_rois):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "loss_labels", utils.SmoothedValue(window_size=100)
    )  # , fmt='{value:.2f}'

    lh_f_pred_all = []
    rh_f_pred_all = []

    for i, samples in tqdm(enumerate(data_loader), total=len(data_loader)):
        samples = tuple(samples.cuda())
        samples = nested_tensor_from_tensor_list(samples)

        outputs = model(samples)

        lh_f_pred = outputs["lh_f_pred"]
        rh_f_pred = outputs["rh_f_pred"]

        if (args.readout_res != "hemis") and (args.readout_res != "voxels"):
            lh_f_pred = outputs["lh_f_pred"][:, :, : args.roi_nums]
            rh_f_pred = outputs["rh_f_pred"][:, :, : args.roi_nums]

            lh_challenge_rois_b = torch.tile(
                lh_challenge_rois[:, :, None], (1, 1, lh_f_pred.shape[0])
            ).permute(2, 1, 0)
            rh_challenge_rois_b = torch.tile(
                rh_challenge_rois[:, :, None], (1, 1, rh_f_pred.shape[0])
            ).permute(2, 1, 0)

            lh_f_pred = torch.sum(torch.mul(lh_challenge_rois_b, lh_f_pred), dim=2)
            rh_f_pred = torch.sum(torch.mul(rh_challenge_rois_b, rh_f_pred), dim=2)

        lh_f_pred_all.append(lh_f_pred.cpu().numpy())
        rh_f_pred_all.append(rh_f_pred.cpu().numpy())

    return np.vstack(lh_f_pred_all), np.vstack(rh_f_pred_all)
