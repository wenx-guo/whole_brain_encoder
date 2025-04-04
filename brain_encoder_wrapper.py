import torch
from torchvision import transforms
import torchvision

from models.activations import get_transformer_activations
from models.brain_encoder import brain_encoder

from datasets.nsd import nsd_dataset_custom, nsd_dataset_avg, nsd_dataset
from engine import evaluate
import numpy as np
from scipy.special import softmax
from utils.args import get_model_dir, get_args_parser, get_default_args
from pathlib import Path, PosixPath
import argparse
import copy
from tqdm import tqdm
from scipy.stats import pearsonr as corr
from huggingface_hub import snapshot_download
import shutil
from PIL import Image


# argparser needs: subj
class BrainEncoderWrapper:
    def __init__(
        self,
        subj=1,
        backbone_arch="dinov2_q",
        encoder_arch="transformer",
        enc_output_layer=[1, 3, 5, 7],
        runs=[1, 2],
    ):
        torch.serialization.add_safe_globals([argparse.Namespace, PosixPath])
        parser = get_args_parser()
        default_args = {
            action.dest: action.default
            for action in parser._actions
            if action.dest != "help"
        }
        args = argparse.Namespace(**default_args)

        self.enc_output_layer = enc_output_layer
        self.runs = runs
        self.subj = subj
        self.default_args = args
        self.neural_data_path = Path(args.data_dir)
        self.parcel_dir = Path(args.parcel_dir)
        self.lr_backbone = None

        self.metadata = np.load(
            Path(args.data_dir) / f"metadata_sub-{self.subj:02}.npy", allow_pickle=True
        ).item()

        self.num_voxels = len(self.metadata["lh_anterior_vertices"]) + len(
            self.metadata["lh_posterior_vertices"]
        )

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(425),
                transforms.CenterCrop(425),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model_paths = {
            "lh": [],
            "rh": [],
        }
        for hemi in ["lh", "rh"]:
            for r in runs:
                for layer_num in self.enc_output_layer:
                    model_path = get_model_dir(
                        args.output_path,
                        backbone_arch,
                        encoder_arch,
                        self.subj,
                        layer_num,
                        r,
                        hemi,
                    )
                    if self.is_valid_model(model_path, hemi):
                        self.model_paths[hemi].append(model_path)

                    else:
                        if not (
                            1 <= self.subj <= 8
                            and r in [1, 2]
                            and layer_num in [1, 3, 5, 7]
                        ):
                            print(f"WARNING: Model path {model_path} is not valid")
                            continue

                        print(
                            f"Downloading checkpoint from huggingface into checkpoints/nsd_test/dinov2_q_transformer/subj_{self.subj:02}/enc_{layer_num}/run_{r}/{hemi}/"
                        )
                        fp = snapshot_download(
                            repo_id="ehwang/brain_encoder_weights",
                            allow_patterns=f"checkpoints/nsd_test/dinov2_q_transformer/subj_{self.subj:02}/enc_{layer_num}/run_{r}/{hemi}/*",
                        )
                        output_path = (
                            Path(args.output_path)
                            / f"nsd_test/dinov2_q_transformer/subj_{self.subj:02}/enc_{layer_num}/run_{r}/{hemi}"
                        )
                        output_path.mkdir(exist_ok=True, parents=True)
                        src_path = (
                            Path(fp)
                            / f"checkpoints/nsd_test/dinov2_q_transformer/subj_{self.subj:02}/enc_{layer_num}/run_{r}/{hemi}"
                        )
                        for item in src_path.rglob(
                            "*"
                        ):  # Recursively match all files and subdirectories
                            dest = output_path / item.relative_to(src_path)
                            if item.is_symlink():
                                # Resolve the symlink to the actual file
                                resolved_path = item.resolve(strict=True)
                                dest.parent.mkdir(
                                    parents=True, exist_ok=True
                                )  # Ensure the parent directory exists
                                shutil.move(str(resolved_path), str(dest))
                            elif item.is_dir():
                                dest.mkdir(parents=True, exist_ok=True)
                            else:
                                dest.parent.mkdir(
                                    parents=True, exist_ok=True
                                )  # Ensure the parent directory exists
                                shutil.move(str(item), str(dest))

                        if self.is_valid_model(model_path, hemi):
                            self.model_paths[hemi].append(model_path)

            assert self.model_paths[hemi], f"No valid models found for {hemi}"

            print(f"Found {len(self.model_paths[hemi])} valid models for {hemi}")

        ## TODO what is the best way to load multiple models?
        val_correlation = {
            "lh": np.zeros((len(self.model_paths[hemi]), self.num_voxels)),
            "rh": np.zeros((len(self.model_paths[hemi]), self.num_voxels)),
        }
        self.corr_sm = copy.deepcopy(val_correlation)
        for hemi in ["lh", "rh"]:
            for idx, model_path in enumerate(self.model_paths[hemi]):
                region_val_corr = np.load(model_path / f"{hemi}_val_corr_nonavg.npy")
                region_val_corr = np.nan_to_num(region_val_corr)
                val_correlation[hemi][idx] += region_val_corr

            self.corr_sm[hemi] = torch.from_numpy(
                softmax(20 * val_correlation[hemi], axis=0)
            )

        print(
            "max lh corr",
            val_correlation["lh"].mean(axis=1).max(),
            "min",
            val_correlation["lh"].mean(axis=1).min(),
        )
        print(
            "max rh corr",
            val_correlation["rh"].mean(axis=1).max(),
            "min",
            val_correlation["rh"].mean(axis=1).min(),
        )

        self.preload_models()

    def is_valid_model(self, model_path, hemi):
        paths = [
            model_path,
            model_path / "checkpoint_nonavg.pth",
            model_path / f"{hemi}_val_corr_nonavg.npy",
        ]
        for p in paths:
            if not p.exists():
                return False

        return True

    def load_model_path(
        self,
        model_path,
        images,
        device="cpu",
    ):
        checkpoint = torch.load(
            model_path / "checkpoint_nonavg.pth", map_location="cpu", weights_only=True
        )

        pretrained_dict = checkpoint["model"]
        args = checkpoint["args"]
        # args.data_dir = self.default_args.data_dir
        # args.imgs_dir = self.default_args.imgs_dir
        # args.parcel_dir = self.default_args.parcel_dir

        dataset = nsd_dataset_avg(args)

        # if "lh" in model_path.name:
        #     args.device = "cuda:0"
        # elif "rh" in model_path.name:
        #     args.device = "cuda:1"
        # else:
        #     raise ValueError("Model path does not contain hemisphere")
        model = brain_encoder(args, dataset)

        if len([k for k in pretrained_dict.keys() if ".orig_mod" in k]) > 0:
            raise ValueError(
                "Model has nonmatching keys with .orig_mod, should manually inspect"
            )

        checkpoint["model"] = {
            key.replace("_orig_mod.", ""): value
            for key, value in checkpoint["model"].items()
        }
        model.load_state_dict(pretrained_dict, strict=False)

        # if "lh" in model_path.name:
        #     print("Loading model to cuda:0")
        #     model = model.to("cuda:0")
        #     model.device = "cuda:0"
        # elif "rh" in model_path.name:
        #     print("Loading model to cuda:1")
        #     model = model.to("cuda:1")
        #     model.device = "cuda:1"
        # else:
        #     raise ValueError("Model path does not contain hemisphere")
        model = model.to(device)
        model.eval()

        return model, args, dataset

    def extract_transformer_features(self, model, imgs, enc_layers=0, dec_layers=1):
        model_features = {}

        outputs, enc_output, enc_attn_weights, dec_output, dec_attn_weights = (
            get_transformer_activations(model, imgs, enc_layers, dec_layers)
        )

        return outputs, enc_output, enc_attn_weights, dec_output, dec_attn_weights

    # def combine_transformer_features(self, model, imgs, runs, enc_output_layers):

    #     for run in self.runs:
    #         for enc_output_layer in self.enc_output_layer:

    #     outputs, enc_output, enc_attn_weights, dec_output, dec_attn_weights = \
    #       self.extract_transformer_features(self, model, imgs)

    def attention(self, images):
        model_features = {}
        dec_attn_weights_all = {"lh": [], "rh": []}

        for hemi in ["lh", "rh"]:
            model_paths = self.model_paths[hemi]
            for idx, model_path in enumerate(
                tqdm(
                    model_paths,
                    desc=f"Running inference on {hemi} models",
                )
            ):
                model, args, imgs_dataset = self.load_model_path(
                    model_path,
                    images,
                    self.device,
                )
                imgs_loader = torch.utils.data.DataLoader(
                    imgs_dataset,
                    batch_size=32,
                    num_workers=4,
                    pin_memory=True,
                    shuffle=False,
                )

                dec_attn_weights_out = []
                for imgs, _ in imgs_loader:
                    imgs = imgs.to(self.device)
                    _, _, _, _, dec_attn_weights = self.extract_transformer_features(
                        model, imgs
                    )
                    dec_attn_weights_out.append(dec_attn_weights[0].detach().cpu())
                dec_attn_weights_out = torch.cat(dec_attn_weights_out, dim=0)

                dec_attn_weights_all[hemi].append(dec_attn_weights_out.cpu().numpy())

                del model

        model_features["dec_attn_weights"] = dec_attn_weights_all

        return model_features

    def forward_hemi(self, hemi, images, use_dataloader):
        hemi_preds = torch.zeros(
            len(images), len(self.model_paths[hemi]), self.num_voxels
        )

        if use_dataloader:
            models = self.models[hemi]
            for idx, model in enumerate(
                tqdm(
                    models,
                    desc=f"Running inference on {hemi} models",
                    leave=False,
                )
            ):
                dataset = nsd_dataset_custom(images, transform=self.transform)
                imgs_loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=32,
                    num_workers=4,
                    pin_memory=True,
                    shuffle=False,
                )
                preds = []
                for imgs, _ in tqdm(
                    imgs_loader, desc="running forward pass", leave=False
                ):
                    # print("imgs", imgs.shape)
                    pred = self.forward_batch(model, imgs)
                    # print("pred", pred.shape)
                    pred = torch.nan_to_num(pred).cpu()
                    preds.append(pred)
                preds = torch.cat(preds, dim=0)
                hemi_preds[:, idx, :] += preds
        else:
            models = self.models[hemi]
            for idx, model in enumerate(models):
                preds = self.forward_batch(model, images)
                preds = torch.nan_to_num(preds).cpu()
                hemi_preds[:, idx, :] += preds

        normalized_pred = (self.corr_sm[hemi].cpu() * hemi_preds.cpu()).sum(1)

        return normalized_pred

    def compile_models(self):
        for hemi in ["lh", "rh"]:
            for idx, model in enumerate(self.models[hemi]):
                self.models[hemi][idx] = torch.compile(model)

    def forward(self, images, use_dataloader=True):
        pred = {
            "lh": np.zeros((len(images), self.num_voxels)),
            "rh": np.zeros((len(images), self.num_voxels)),
        }

        for hemi in ["lh", "rh"]:
            pred[hemi] = self.forward_hemi(hemi, images, use_dataloader)

        return pred

    def preload_models(self):
        self.models = {}

        for hemi in ["lh", "rh"]:
            self.models[hemi] = []

            for model_path in self.model_paths[hemi]:
                model, _, _ = self.load_model_path(
                    model_path,
                    torch.zeros(1, 3, 224, 224),
                    self.device,
                )
                self.models[hemi].append(model)

    def forward_batch(self, model, images):
        if self.lr_backbone is not None:
            model.lr_backbone = self.lr_backbone
            for name, param in model.named_parameters():
                param.requires_grad = False

        imgs = images.to(next(model.parameters()).device, non_blocking=True)
        outputs = model(imgs)
        outputs = outputs["pred"]

        return outputs

    def load_roi_labels(self):
        metadata = np.load(
            self.neural_data_path / f"metadata_sub-{self.subj:02}.npy",
            allow_pickle=True,
        ).item()

        return {"lh": metadata["lh_rois"], "rh": metadata["rh_rois"]}

    def load_parcels(self):
        parcels = {}
        for hemi in ["lh", "rh"]:
            parcels[hemi] = torch.load(
                self.parcel_dir / f"{hemi}_labels_s{self.subj:02}.pt", weights_only=True
            )

        return parcels


def main():
    torch.serialization.add_safe_globals([argparse.Namespace, PosixPath])

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--subj", type=int, default=1)
    argparser.add_argument(
        "--results_dir",
        type=str,
        default="/engram/nklab/algonauts/ethan/whole_brain_encoder/results",
    )
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--target_dir", type=str, default=None)
    argparser.add_argument("--save_path", type=str, default=None)
    argparser.add_argument("--exist_skip", type=bool, default=False)
    input_args = argparser.parse_args()

    if input_args.split == "folder":
        assert input_args.target_dir is not None
        print(f"Running inference on images in {input_args.target_dir}")

        model = BrainEncoderWrapper(
            subj=input_args.subj,
            enc_output_layer=[1, 3, 5, 7],
            runs=[1, 2],
        )

        args = get_default_args()
        target_dir = Path(input_args.target_dir)

        if (target_dir / "activations.npy").exists() and input_args.exist_skip:
            print(f"Activations already exist for {target_dir}, skipping")
            return

        imgs_data = []
        img_paths = []
        for img_file in sorted(target_dir.glob("*")):
            if img_file.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                continue
            img_paths.append(img_file)
            image = Image.open(img_file).convert("RGB")
            imgs_data.append(image)
        # imgs_data.append(np.array(image))
        # print(np.array(image).shape)
        # imgs_data = np.stack(imgs_data)
        # print(f"image[0] shape: {np.array(imgs_data[0]).shape}")

        dataset = nsd_dataset_custom(imgs_data, transform=model.transform)
        batch_size = 16
        imgs_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
        )

        out = {
            "lh": np.zeros((len(imgs_data), 163842)),
            "rh": np.zeros((len(imgs_data), 163842)),
        }
        for idx, (imgs, _) in tqdm(
            enumerate(imgs_loader),
            desc="running forward pass",
            leave=False,
            total=len(imgs_loader),
        ):
            with torch.no_grad():
                pred = model.forward(imgs, use_dataloader=False)

            for hemi in ["lh", "rh"]:
                out[hemi][idx * batch_size : idx * batch_size + len(imgs)] = (
                    pred[hemi].cpu().numpy()
                )

        parcels = model.load_parcels()
        roi_labels = model.load_roi_labels()
        parcel_mean_activity = {}
        for hemi in ["lh", "rh"]:
            parcel_mean_activity[hemi] = np.zeros((len(imgs_data), len(parcels[hemi])))
            for idx, parcel in enumerate(parcels[hemi]):
                parcel_mask = np.zeros(163842, dtype=bool)
                parcel_mask[parcel] = True
                parcel_mean_activity[hemi][:, idx] = out[hemi][:, parcel_mask].mean(
                    axis=1
                )
            la = np.zeros(163842, dtype=bool)
            for roi in roi_labels[hemi]:
                la = np.logical_or(la, roi_labels[hemi][roi])
            out[hemi] = out[hemi][:, la]

        res = {}
        res["path"] = input_args.target_dir
        res["out"] = out
        res["parcel_mean_activity"] = parcel_mean_activity
        res["img_paths"] = img_paths
        res["parcels"] = parcels

        if input_args.save_path is not None and input_args.save_path:
            save_path = Path(input_args.save_path)
        else:
            save_path = target_dir / "activations.npy"
        save_path.parent.mkdir(exist_ok=True, parents=True)
        np.save(save_path, res)
        print(f"Saved activations to {save_path}")

    if input_args.split in ["train", "val", "test"]:
        split = input_args.split

        model = BrainEncoderWrapper(
            subj=input_args.subj,
            enc_output_layer=[1, 3, 5, 7],
            runs=[1, 2],
        )

        save_dir = (
            Path(input_args.results_dir)
            / f"enc_{'_'.join([str(s) for s in model.enc_output_layer])}_run_{'_'.join([str(s) for s in model.runs])}"
            / f"subj_{input_args.subj:02}"
        )

        print(f"Saving results to {save_dir}")

        args = get_default_args()
        args.subj = input_args.subj
        imgs = {}
        betas = {}
        for hemi in ["lh", "rh"]:
            args.hemi = hemi
            val_dataset = nsd_dataset_avg(args, transform=None, split=input_args.split)
            val_dataset.backbone_arch = False

            imgs[hemi] = []
            betas[hemi] = []
            for img, beta in val_dataset:
                imgs[hemi].append(img)
                betas[hemi].append(beta["betas"])
            imgs[hemi] = np.stack(imgs[hemi])
            betas[hemi] = np.stack(betas[hemi])
        imgs = imgs["lh"]

        out = model.forward(imgs_data)
        for key, t in out.items():
            out[key] = t.cpu().numpy()
        res = {}
        res["out"] = out

        save_dir.mkdir(exist_ok=True, parents=True)

        np.save(save_dir / f"{split}_activations.npy", res)

        val_correlation = {}
        for hemi in ["lh", "rh"]:
            ys = torch.from_numpy(betas[hemi])

            num_valid_voxels = ys.shape[1]
            val_correlation[hemi] = torch.zeros(num_valid_voxels)
            for v in range(num_valid_voxels):
                val_correlation[hemi][v] = corr(ys[:, v], out[hemi][:, v])[0]

            val_correlation[hemi] = val_correlation[hemi].numpy()

            print(
                f"Validation correlation for {hemi} hemi: {val_correlation[hemi].mean()}"
            )

            with (save_dir / f"{split}_corr_avg.txt").open("a") as f:
                f.write(
                    f"Validation correlation for {split} split: {val_correlation[hemi].mean()}\n"
                )

        np.save(save_dir / f"{split}_corr_avg.npy", val_correlation)


if __name__ == "__main__":
    main()
