import torch
from torchvision import transforms

# from models.activations import get_transformer_activations
from models.brain_encoder import brain_encoder

from datasets.nsd import nsd_dataset_custom, nsd_dataset_avg
from engine import evaluate
import numpy as np
from scipy.special import softmax
from utils.args import get_model_dir, get_args_parser, get_default_args
from pathlib import Path, PosixPath
import argparse
import copy
from tqdm import tqdm
from scipy.stats import pearsonr as corr


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

        self.metadata = np.load(
            Path(args.data_dir) / f"metadata_sub-{self.subj:02}.npy", allow_pickle=True
        ).item()

        self.num_voxels = len(self.metadata["lh_anterior_vertices"]) + len(
            self.metadata["lh_posterior_vertices"]
        )

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
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
                        print(f"WARNING: Model path {model_path} is not valid")

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
        args.data_dir = self.default_args.data_dir
        args.imgs_dir = self.default_args.imgs_dir
        args.parcel_dir = self.default_args.parcel_dir
        args.data_dir = self.default_args.data_dir

        dataset = nsd_dataset_custom(images, args, transform=self.transform)

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

        model = model.to(device)
        model.eval()

        return model, args, dataset

    # def extract_transformer_features(self, model, imgs, enc_layers=0, dec_layers=1):
    #     model_features = {}
    #     try:
    #         model = model.module
    #     except:
    #         model = model

    #     outputs, enc_output, enc_attn_weights, dec_output, dec_attn_weights = (
    #         get_transformer_activations(model, imgs, enc_layers, dec_layers)
    #     )

    #     return outputs, enc_output, enc_attn_weights, dec_output, dec_attn_weights

    # def combine_transformer_features(self, model, imgs, runs, enc_output_layers):

    #     for run in self.runs:
    #         for enc_output_layer in self.enc_output_layer:

    #     outputs, enc_output, enc_attn_weights, dec_output, dec_attn_weights = \
    #       self.extract_transformer_features(self, model, imgs)

    # def attention(self, images):
    #     # images = images.to(self.device)
    #     model_features = {}
    #     if self.model is not None:
    #         outputs, enc_output, enc_attn_weights, dec_output, dec_attn_weights = (
    #             self.extract_transformer_features(self.model, images)
    #         )

    #         # print('dec_attn_weights', len(dec_attn_weights), dec_attn_weights[0].shape)

    #         # model_features['outputs'] = outputs
    #         # model_features['enc_output'] = enc_output
    #         # model_features['enc_attn_weights'] = enc_attn_weights
    #         # model_features['dec_output'] = dec_output
    #         model_features["dec_attn_weights"] = dec_attn_weights

    #     else:
    #         dec_attn_weights_all = []
    #         for enc_output_layer in self.enc_output_layer:
    #             for run in self.runs:
    #                 print(f"Run {run}")
    #                 # subj = format(self.subj, '02')
    #                 model_path = f"{self.results_dir}/nsd_test/{self.arch}/subj_{self.subj}/{self.readout_res}/enc_{enc_output_layer}/run_{run}/"
    #                 model, _ = self.load_model_path(model_path)

    #                 _, _, _, _, dec_attn_weights = self.extract_transformer_features(
    #                     model, images.to(self.device)
    #                 )

    #                 dec_attn_weights_all.append(
    #                     dec_attn_weights[0].detach().cpu().numpy()
    #                 )

    #                 del model

    #         model_features["dec_attn_weights"] = dec_attn_weights_all

    #     return model_features

    def forward(self, images):
        pred = {
            "lh": np.zeros((images.shape[0], self.num_voxels)),
            "rh": np.zeros((images.shape[0], self.num_voxels)),
        }

        for hemi in ["lh", "rh"]:
            hemi_preds = torch.zeros(
                len(images), len(self.model_paths[hemi]), self.num_voxels
            )
            model_paths = self.model_paths[hemi]

            for idx, model_path in enumerate(
                tqdm(
                    model_paths,
                    desc=f"Running inference on {hemi} models",
                )
            ):
                preds = self.forward_region(model_path, images)
                preds = torch.nan_to_num(preds)
                hemi_preds[:, idx, :] += preds

            pred[hemi] = (
                self.corr_sm[hemi].to(self.device) * hemi_preds.to(self.device)
            ).sum(1)

        return pred

    def forward_region(self, model_path, images):
        model, args, imgs_dataset = self.load_model_path(
            model_path,
            images,
            self.device,
        )

        imgs_loader = torch.utils.data.DataLoader(
            imgs_dataset,
            batch_size=16,
            num_workers=4,
            pin_memory=True,
        )

        criterion = None
        output, _ = evaluate(
            args, model, criterion, imgs_loader, imgs_dataset, print_freq=10
        )

        return output


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def main():
    torch.serialization.add_safe_globals([argparse.Namespace, PosixPath])

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--subj", type=int, default=1)
    argparser.add_argument(
        "--results_dir",
        type=str,
        default="/engram/nklab/algonauts/ethan/whole_brain_encoder/results",
    )
    input_args = argparser.parse_args()

    model = BrainEncoderWrapper(
        subj=input_args.subj,
        enc_output_layer=[1, 3, 5, 7],
        runs=[1, 2],
        results_dir=input_args.results_dir,
    )

    save_dir = (
        Path(model.results_dir)
        / f"enc_{'_'.join([str(s) for s in model.enc_output_layer])}_run_{'_'.join([str(s) for s in model.runs])}"
        / f"subj_{input_args.subj:02}"
    )

    print(f"Saving results to {save_dir}")

    args = get_default_args()
    args.subj = input_args.subj
    args.metaparcel_idx = 0
    val_dataset = nsd_dataset_avg(args, transform=None, split="test")
    imgs_data = []
    for img, _ in val_dataset:
        imgs_data.append(img)
    imgs_data = np.stack(imgs_data)

    out = model.forward(imgs_data)

    split = "test"
    save_dir.mkdir(exist_ok=True, parents=True)
    val_correlation = {}
    for hemi in ["lh", "rh"]:
        args.metaparcel_idx = 0
        args.hemi = hemi
        val_dataset = nsd_dataset_avg(args, transform=None, split=split)

        data_idxs = [val_dataset.img_to_runs[i] for i in range(len(val_dataset))]
        data = [
            torch.from_numpy(val_dataset.betas[idxs]).mean(axis=0) for idxs in data_idxs
        ]
        ys = torch.stack(data)

        num_valid_voxels = ys.shape[1]
        val_correlation[hemi] = torch.zeros(num_valid_voxels)
        for v in range(num_valid_voxels):
            val_correlation[hemi][v] = corr(ys[:, v].cpu(), out[hemi][:, v].cpu())[0]

        val_correlation[hemi] = val_correlation[hemi].numpy()

        print(f"Validation correlation for {hemi} hemi: {val_correlation[hemi].mean()}")

        with (save_dir / f"{split}_corr_avg.txt").open("a") as f:
            f.write(
                f"Validation correlation for {split} split: {val_correlation[hemi].mean()}\n"
            )

    np.save(save_dir / f"{split}_corr_avg.npy", val_correlation)


if __name__ == "__main__":
    main()
