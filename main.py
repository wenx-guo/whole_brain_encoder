import os
import argparse

from tqdm import tqdm

import torch
import torch.utils.model_zoo
from torchvision import transforms

from scipy.stats import pearsonr as corr

from models.brain_encoder import brain_encoder
from engine import evaluate as evaluate
from engine import train_one_epoch as train_one_epoch

import utils.utils as utils
from pathlib import Path

from datasets.nsd import nsd_dataset as nsd_dataset
from datasets.nsd import nsd_dataset_avg

import wandb

torch.manual_seed(0)

os.environ["WANDB_MODE"] = "offline"


def get_args_parser():
    parser = argparse.ArgumentParser(description="NSD Training", add_help=False)

    parser.add_argument("--resume", default=None, help="resume from checkpoint")
    parser.add_argument(
        "--output_path",
        default="/engram/nklab/algonauts/ethan/transformer_brain_encoder/checkpoints",
        type=str,
        help="if not none, then store the model resuls",
    )

    parser.add_argument("--save_model", default=True, type=int)

    ## NSD params
    parser.add_argument("--subj", default=1, type=int)
    parser.add_argument("--run", default=1, type=int)
    parser.add_argument(
        "--data_dir",
        default="/engram/nklab/datasets/natural_scene_dataset/model_training_datasets/neural_data",
        type=str,
    )
    parser.add_argument(
        "--imgs_dir",
        default="/engram/nklab/datasets/natural_scene_dataset/nsddata_stimuli/stimuli/nsd",
        type=str,
    )
    parser.add_argument(
        "--parcel_dir",
        default="/engram/nklab/algonauts/ethan/parcelling/results/200c_20percentile_ftalgo_5init_3000iter_train",
        type=str,
    )
    parser.add_argument(
        "--parent_submission_dir",
        default="./algonauts_2023_challenge_submission/",
        type=str,
    )

    parser.add_argument("--saved_feats", default=None, type=str)  #'dinov2q'
    parser.add_argument(
        "--saved_feats_dir", default="../../algonauts_image_features/", type=str
    )

    parser.add_argument(
        "--readout_res",
        choices=[
            "voxels",
            "streams_inc",
            "visuals",
            "bodies",
            "faces",
            "places",
            "words",
            "hemis",
            "parcels",
        ],  # TODO: add clusters
        default="parcels",
        type=str,
    )

    # the model for mapping from backbone image features to fMRI
    parser.add_argument(
        "--encoder_arch",
        choices=["transformer", "linear"],
        default="transformer",
        type=str,
    )

    parser.add_argument(
        "--objective",
        choices=["NSD"],
        default="classification",
        help="which model to train",
    )

    # Backbone
    parser.add_argument(
        "--backbone_arch",
        choices=[
            None,
            "dinov2",
            "dinov2_q",
            "resnet18",
            "resnet50",
            "dinov2_special_token",
            "dinov2_q_special_token",
        ],
        default="dinov2_q",
        type=str,
        help="Name of the backbone to use",
    )  # resnet50 resnet18 dinov2

    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )
    parser.add_argument(
        "--return_interm",
        default=False,
        help="Train segmentation head if the flag is provided",
    )

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )

    # * Transformer
    parser.add_argument(
        "--enc_layers",
        default=0,
        type=int,
        help="Number of encoding layers in the transformer brain model",
    )
    parser.add_argument(
        "--dec_layers",
        default=1,
        type=int,
        help="Number of decoding layers in the transformer brain model",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=1024,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=768,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )  # 256  #868 (100+768)
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=16,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument(
        "--num_queries", default=16, type=int, help="Number of query slots"
    )
    parser.add_argument("--pre_norm", action="store_true")

    parser.add_argument(
        "--enc_output_layer",
        default=1,
        type=int,
        help="Specify the encoder layer that provides the encoder output. default is the last layer",
    )

    # training parameters
    parser.add_argument(
        "--num_workers", default=4, type=int, help="number of data loading num_workers"
    )
    parser.add_argument(
        "--epochs", default=30, type=int, help="number of total epochs to run"
    )
    parser.add_argument("--batch_size", default=16, type=int, help="mini-batch size")
    parser.add_argument(
        "--lr", default=0.0005, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--weight_decay", default=1e-4, type=float, help="weight decay "
    )
    parser.add_argument("--lr_drop", default=4, type=int)
    parser.add_argument("--lr_backbone", default=0, type=int)
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )

    parser.add_argument("--evaluate", action="store_true", help="just evaluate")

    parser.add_argument("--wandb_p", default="brain_encoder", type=str)
    parser.add_argument("--wandb_r", default=None, type=str)

    # dataset parameters
    parser.add_argument(
        "--image_size",
        default=None,
        type=int,
        help="what size should the image be resized to?",
    )
    parser.add_argument(
        "--horizontal_flip",
        default=True,
        help="whether to use horizontal flip augmentation",
    )

    parser.add_argument(
        "--img_channels",
        default=3,
        type=int,
        help="what should the image channels be (not what it is)?",
    )  # gray scale 1 / color 3

    parser.add_argument("--lh_vs", default=None)
    parser.add_argument("--rh_vs", default=None)

    parser.add_argument(
        "--axis", default="anterior", choices=["anterior", "posterior"], type=str
    )
    parser.add_argument("--hemi", default="lh", choices=["lh", "rh"], type=str)

    return parser


def main(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", args.device)

    args.subj = format(args.subj, "02")
    args.subject_submission_dir = os.path.join(
        args.parent_submission_dir, "subj" + args.subj
    )

    if args.axis == "posterior":
        args.metaparcel_idx = 0
    elif args.axis == "anterior":
        args.metaparcel_idx = 1

    if args.output_path:
        args.output_path = Path(args.output_path)
        args.save_dir = Path(
            args.output_path
            / f"nsd_test/{args.backbone_arch}_{args.encoder_arch}/subj_{args.subj}/{args.hemi}_{args.axis}"
        )
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

        wandb.init(
            project=args.wandb_p,
            name=f"{args.hemi} {args.axis} sub{args.subj} {wandb_r}",
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
        num_workers=16,
        pin_memory=True,
    )

    val_dataset = nsd_dataset(args, transform=transform, split="val")
    val_dataset_avg = nsd_dataset_avg(args, transform=transform, split="val")
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        num_workers=8,
        pin_memory=True,
    )
    val_loader_avg = torch.utils.data.DataLoader(
        val_dataset_avg,
        batch_size=32,
        num_workers=8,
        pin_memory=True,
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

            ys, preds = evaluate(
                args,
                model,
                criterion,
                dl,
                train_dataset,
            )

            p = torch.tensor(
                [corr(ys[i].detach().cpu(), preds[i].cpu()) for i in range(ys.shape[0])]
            )
            val_perf[dataset_type] = p.mean(axis=0)[0].item()

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

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "model training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_path:
        Path(args.output_path).mkdir(parents=True, exist_ok=True)

    # TODO: fix the shuffling issue before enabling distributed training
    # if args.distributed:
    #     args.world_size = torch.cuda.device_count()
    #     mp.spawn(main, args=(args.world_size, args), nprocs=args.world_size)
    # else:
    main(args)
