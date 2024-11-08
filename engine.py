import math
import sys
from typing import Iterable
import torch
from tqdm import tqdm

from utils.utils import NestedTensor, nested_tensor_from_tensor_list

import utils.utils as utils
import numpy as np
import wandb


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_valid_voxels,
    dataset,
    max_norm: float = 0,
):
    model.train()
    print_freq = 10
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", utils.SmoothedValue(window_size=print_freq, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "loss_labels", utils.SmoothedValue(window_size=print_freq)
    )  # , fmt='{value:.2f}'
    header = "Epoch: [{}]".format(epoch)

    mask = dataset.get_parcel_mask(dataset.parcels["lh_betas"][0])
    num_valid_voxels = mask.sum()

    running_loss = 0
    running_corr = 0

    for batch_idx, (imgs, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets["lh_betas"][0].to(device, non_blocking=True).to(torch.float32)
        outputs = model(imgs)
        outputs = outputs["pred"]
        # loss = criterion(outputs, targets) / num_valid_voxels
        loss = criterion(outputs, targets) / num_valid_voxels

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

        out = unwrap_fmri(outputs.shape[0], outputs.cpu(), mask, dataset)
        y = unwrap_fmri(targets.shape[0], targets.cpu(), mask, dataset)

        train_corr = torch.corrcoef(torch.stack([out.flatten(), y.flatten()]))[
            0, 1
        ].item()

        running_loss += loss.item()
        running_corr += train_corr
        if batch_idx % print_freq == print_freq - 1:
            wandb.log(
                {
                    "Training Loss": running_loss / print_freq,
                    "Epoch": epoch,
                    "Training Corr": running_corr / print_freq,
                    "Batch": batch_idx + (epoch + 1) * print_freq,
                }
            )
            running_loss = 0
            running_corr = 0

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_exp(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_valid_voxels,
    dataset,
    max_norm: float = 0,
):
    model.train()
    print_freq = 20
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", utils.SmoothedValue(window_size=print_freq, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "loss_labels", utils.SmoothedValue(window_size=print_freq)
    )  # , fmt='{value:.2f}'
    header = "Epoch: [{}]".format(epoch)

    # mask = dataset.get_parcel_mask(dataset.parcels["lh_betas"][0])
    # num_valid_voxels = mask.sum()
    num_valid_voxels = dataset.metadata["lh_rois"]["V1v"].sum()

    running_loss = 0
    running_corr = 0

    for batch_idx, (imgs, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        imgs = imgs.to(device, non_blocking=True)
        targets = (
            targets["lh_raw"][:, :num_valid_voxels]
            .to(device, non_blocking=True)
            .to(torch.float32)
        )
        outputs = model(imgs)
        outputs = outputs["pred"]
        # loss = criterion(outputs, targets) / num_valid_voxels
        loss = criterion(outputs, targets) / num_valid_voxels

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

        train_corr = torch.corrcoef(
            torch.stack([targets.flatten(), outputs.flatten()])
        )[0, 1].item()

        running_loss += loss.item()
        running_corr += train_corr
        if batch_idx % print_freq == print_freq - 1:
            wandb.log(
                {
                    "Training Loss": running_loss / print_freq,
                    "Epoch": epoch,
                    "Training Corr": running_corr / print_freq,
                    "Batch": batch_idx + (epoch + 1) * print_freq,
                }
            )
            running_loss = 0
            running_corr = 0

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def unwrap_fmri(batch_size, fmri_data, mask, dataset):
    recon = torch.zeros(batch_size, *dataset.labels["lh_betas"][:, 0].shape)
    for idxs, betas, m in zip(
        dataset.parcels["lh_betas"][0], fmri_data.permute(1, 0, 2), mask
    ):
        recon[:, idxs] = betas[:, m]
    return recon[:, dataset.labels["lh_betas"][:, 0] == 0]


@torch.no_grad()
def evaluate(model, criterion, data_loader, args, dataset, device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "loss_labels", utils.SmoothedValue(window_size=100)
    )  # , fmt='{value:.2f}'
    header = "Test:"

    preds = []
    ys = []

    mask = dataset.get_parcel_mask(dataset.parcels["lh_betas"][0])
    num_valid_voxels = mask.sum()

    for imgs, targets in metric_logger.log_every(data_loader, 25, header):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets["lh_betas"][0].to(device, non_blocking=True).to(torch.float32)
        outputs = model(imgs)
        outputs = outputs["pred"]
        loss = criterion(outputs, targets) / num_valid_voxels

        preds.append(unwrap_fmri(outputs.shape[0], outputs.cpu(), mask, dataset))
        ys.append(unwrap_fmri(targets.shape[0], targets.cpu(), mask, dataset))

        # ys.append(targets.flatten(start_dim=1)[:, mask.flatten()])
        # preds.append(outputs.flatten(start_dim=1)[:, mask.flatten()])

        loss_value = loss.item()
        metric_logger.update(loss_labels=loss_value)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return torch.vstack(ys), torch.vstack(preds)


@torch.no_grad()
def evaluate_exp(model, criterion, data_loader, args, dataset, device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "loss_labels", utils.SmoothedValue(window_size=100)
    )  # , fmt='{value:.2f}'
    header = "Test:"

    preds = []
    ys = []
    corrs = []

    mask = dataset.get_parcel_mask(dataset.parcels["lh_betas"][0])
    num_valid_voxels = mask.sum()
    num_valid_voxels = dataset.metadata["lh_rois"]["V1v"].sum()

    for imgs, targets in metric_logger.log_every(data_loader, 25, header):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets["lh_v1v"].to(device, non_blocking=True).to(torch.float32)
        outputs = model(imgs)
        outputs = outputs["pred"]
        loss = criterion(outputs, targets) / num_valid_voxels

        # preds.append(unwrap_fmri(outputs.shape[0], outputs.cpu(), mask, dataset))
        # ys.append(unwrap_fmri(targets.shape[0], targets.cpu(), mask, dataset))
        train_corr = torch.corrcoef(
            torch.stack([targets.flatten(), outputs.flatten()])
        )[0, 1].item()

        ys.append(targets)
        preds.append(outputs)
        corrs.append(train_corr)

        loss_value = loss.item()
        metric_logger.update(loss_labels=loss_value)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # return torch.vstack(ys), torch.vstack(preds)
    return np.mean(corrs)


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
                                                                                          