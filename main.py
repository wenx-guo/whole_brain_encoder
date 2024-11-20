import os
import argparse

from tqdm import tqdm
import torch
import torch.utils.model_zoo
from torchvision import transforms

import numpy as np

from scipy.stats import pearsonr as corr

from models.brain_encoder import brain_encoder
from engine import evaluate as evaluate
from engine import train_one_epoch as train_one_epoch

from utils.args import get_args_parser, get_model_dir
import utils.utils as utils

from pathlib import Path

from datasets.nsd import nsd_dataset as nsd_dataset
from datasets.nsd import nsd_dataset_avg

import wandb

os.environ["WANDB_MODE"] = "offline"


def main(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", args.device)

    args.subj = format(args.subj, "02")
    args.subject_submission_dir = os.path.join(
        args.parent_submission_dir, "subj" + args.subj
    )

    if args.axis == "posterior":
        args.metaparcel_idx = 1
    elif args.axis == "anterior":
        args.metaparcel_idx = 0

    if args.output_path:
        args.save_dir = get_model_dir(args)

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir, exist_ok=True)
        print("saving into:", args.save_dir)

    if args.wandb_p:
        os.environ["WANDB_MODE"] = "online"

        if args.wandb_r:
            wandb_r = args.wandb_r
        else:
            wandb_r = args.encoder_arch

        os.environ["WANDB__SERVICE_WAIT"] = "300"

        args.node_name = os.getenv("SLURMD_NODENAME")
        args.job_id = os.getenv("SLURM_JOB_ID")  # Job ID
        args.cpus = os.getenv(
            "SLURM_CPUS_ON_NODE"
        )  # Number of CPUs allocated on this node
        args.mem = os.getenv(
            "SLURM_MEM_PER_NODE"
        )  # Memory per node in MB (may not always be set)
        args.gpus = os.getenv("SLURM_GPUS")

        wandb.init(
            project=args.wandb_p,
            name=f"sub{args.subj} {args.hemi} {args.axis} {args.enc_output_layer} {wandb_r} {args.run}",
            config={
                "learning_rate": args.lr,
                "architecture": f"{args.encoder_arch}",
                "epochs": args.epochs,
                "subject": args.subj,
                "lr": args.lr,
                **vars(args),
            },
        )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # convert the images to a PyTorch tensor
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),  # normalize the images color channels
        ]
    )
    train_dataset = nsd_dataset(args, transform=transform, preload_data=True)
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    val_dataset = nsd_dataset(args, transform=transform, split="val")
    val_dataset_avg = nsd_dataset_avg(args, transform=transform, split="val")
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=16,
        num_workers=4,
        # pin_memory=True,
        # persistent_workers=True,
    )
    val_loader_avg = torch.utils.data.DataLoader(
        val_dataset_avg,
        batch_size=16,
        num_workers=4,
        # pin_memory=True,
        # persistent_workers=True,
    )
    print(
        f"len train_loader: {len(trainloader)}, len val_loader: {len(val_loader)}, len val_loader_avg: {len(val_loader_avg)}"
    )
    model = brain_encoder(args, train_dataset)
    model = model.cuda()
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of model parameters: {num_parameters}")
    print("linear layer weights shape:", model.embed.shape)
    print(model)
    model = torch.compile(model)

    criterion = torch.nn.MSELoss(reduction="sum")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        pretrained_dict = checkpoint["model"]
        model.load_state_dict(pretrained_dict)

        args.best_val_acc = vars(checkpoint["args"])[
            "val_perf"
        ]  # checkpoint['val_acc'] #or read it from the

        if (
            not args.eval
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            train_params = checkpoint["train_params"]
            param_dicts = [
                {
                    "params": [
                        p for n, p in model.named_parameters() if n in train_params
                    ]
                },
            ]

            optimizer = torch.optim.AdamW(
                param_dicts, lr=args.lr, weight_decay=args.weight_decay
            )
            optimizer.load_state_dict(checkpoint["optimizer"])

            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, args.lr_drop, gamma=0.5
            )
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, args.lr_drop, gamma=0.5
        )
        torch.set_float32_matmul_precision("high")

    print("Start training")
    best_val_perf = {"nonavg": 0, "avg": 0}
    for epoch in range(args.start_epoch, args.epochs):
        _ = train_one_epoch(
            args,
            model,
            criterion,
            trainloader,
            optimizer,
            epoch,
            train_dataset,
            args.clip_max_norm,
        )

        lr_scheduler.step()

        val_perf = {"nonavg": 0, "avg": 0}

        for dataset_type, dl in zip(["nonavg", "avg"], [val_loader, val_loader_avg]):
            print(f"evaluating {dataset_type} dataset")

            val_correlation = evaluate(
                args,
                model,
                criterion,
                dl,
                train_dataset,
            )

            val_perf[dataset_type] = val_correlation.mean().item()

            print(f"{dataset_type} val_perf:", val_perf[dataset_type])

            if args.wandb_p:
                wandb.log(
                    {f"val_perf_{dataset_type}": val_perf[dataset_type], "epoch": epoch}
                )

            if args.output_path:
                # update best validation acc and save best model to output dir
                if val_perf[dataset_type] > best_val_perf[dataset_type]:
                    best_val_perf[dataset_type] = val_perf[dataset_type]

                    with open(args.save_dir / "val_results.txt", "a") as f:
                        f.write(
                            f"epoch {epoch}, {dataset_type} val_perf: {val_perf[dataset_type]} \n"
                        )

                    try:
                        if args.save_model:
                            checkpoint_paths = [
                                args.save_dir / f"checkpoint_{dataset_type}.pth"
                            ]
                            # print('checkpoint_path:',  checkpoint_paths)
                            for checkpoint_path in checkpoint_paths:
                                utils.save_on_master(
                                    {
                                        "model": model.state_dict(),
                                        "optimizer": optimizer.state_dict(),
                                        "lr_scheduler": lr_scheduler.state_dict(),
                                        "epoch": epoch,
                                        "args": args,
                                        "val_perf": best_val_perf[dataset_type],
                                        "dataset_type": dataset_type,
                                    },
                                    checkpoint_path,
                                )
                    except Exception as e:
                        print("Error saving model")
                        print(e)

                    np.save(
                        args.save_dir / f"{args.hemi}_val_corr.npy",
                        val_correlation.numpy(),
                    )

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "model training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_path:
        Path(args.output_path).mkdir(parents=True, exist_ok=True)

    main(args)
