import torch
from torchvision import transforms

# from models.activations import get_transformer_activations
from models.brain_encoder import brain_encoder

# from datasets.nsd_utils import roi_maps, roi_masks
from datasets.nsd import nsd_dataset_custom
from engine import evaluate
import numpy as np
from scipy.special import softmax
from utils.args import get_model_dir, get_args_parser
from pathlib import Path
import argparse
import copy
from tqdm import tqdm


class BrainEncoderWrapper:
    def __init__(
        self,
        subj=1,
        backbone_arch="dinov2_q",
        encoder_arch="transformer",
        enc_output_layer=[1],
        runs=[1],
        results_dir=None,
    ):
        parser = get_args_parser()
        default_args = {
            action.dest: action.default
            for action in parser._actions
            if action.dest != "help"
        }
        args = argparse.Namespace(**default_args)

        self.enc_output_layer = enc_output_layer  # 1
        self.subj = format(subj, "02")

        self.metadata = np.load(
            Path(args.data_dir) / f"metadata_sub-{self.subj:02}.npy", allow_pickle=True
        ).item()

        self.num_voxels = len(self.metadata["lh_anterior_vertices"]) + len(
            self.metadata["lh_posterior_vertices"]
        )

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),  # convert the images to a PyTorch tensor
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),  # normalize the images color channels
            ]
        )

        if results_dir is None:
            self.results_dir = Path(
                "/engram/nklab/algonauts/ethan/transformer_brain_encoder/results"
            )

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # data_dir = "/engram/nklab/algonauts/algonauts_2023_challenge_data/"
        # self.data_dir = os.path.join(data_dir, "subj" + self.subj)
        # /engram/nklab/hossein/recurrent_models/transformer_brain_encoder/results/

        # roi_name_maps, lh_challenge_rois, rh_challenge_rois = roi_maps(self.data_dir)
        # (
        #     self.lh_challenge_rois,
        #     self.rh_challenge_rois,
        #     self.lh_roi_names,
        #     self.rh_roi_names,
        #     num_queries,
        # ) = roi_masks(
        #     self.readout_res, roi_name_maps, lh_challenge_rois, rh_challenge_rois
        # )

        self.model_paths = {
            "lh": {"anterior": [], "posterior": []},
            "rh": {"anterior": [], "posterior": []},
        }
        for hemi in ["lh", "rh"]:
            for axis in ["anterior", "posterior"]:
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
                            axis,
                        )
                        if self.is_valid_model(model_path, hemi):
                            self.model_paths[hemi][axis].append(model_path)

                        else:
                            print(f"WARNING: Model path {model_path} is not valid")

                assert self.model_paths[hemi], f"No valid models found for {hemi}"
                assert self.model_paths[hemi][
                    axis
                ], f"No valid models found for {hemi} {axis}"

                print(
                    f"Found {len(self.model_paths[hemi][axis])} valid models for {hemi} {axis}"
                )

        ## TODO what is the best way to load multiple models?
        val_correlation = {
            "lh": np.zeros((len(self.model_paths[hemi][axis]), self.num_voxels)),
            "rh": np.zeros((len(self.model_paths[hemi][axis]), self.num_voxels)),
        }
        self.corr_sm = copy.deepcopy(val_correlation)
        for hemi in ["lh", "rh"]:
            for axis in ["anterior", "posterior"]:
                for idx, model_path in enumerate(self.model_paths[hemi][axis]):
                    region_val_corr = np.load(
                        model_path / f"{hemi}_val_corr_nonavg.npy"
                    )
                    region_val_corr = np.nan_to_num(region_val_corr)
                    val_correlation[hemi][idx] += region_val_corr

            self.corr_sm[hemi] = torch.from_numpy(
                softmax(20 * val_correlation[hemi], axis=0)
            )

    def is_valid_model(self, model_path, hemi):
        paths = [
            model_path,
            model_path / "checkpoint_nonavg.pth",
            model_path / f"{hemi}_val_corr_nonavg.npy",
            model_path / f"{hemi}_val_corr_avg.npy",
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
            model_path / "checkpoint_nonavg.pth", map_location="cpu"
        )

        pretrained_dict = checkpoint["model"]
        args = checkpoint["args"]

        dataset = nsd_dataset_custom(images, args, transform=self.transform)

        model = brain_encoder(args, dataset)

        checkpoint = torch.load(
            model_path / "checkpoint_nonavg.pth", map_location="cpu", weights_only=False
        )
        checkpoint["model"] = {
            key.replace("_orig_mod.", ""): value
            for key, value in checkpoint["model"].items()
        }
        pretrained_dict = checkpoint["model"]
        model.load_state_dict(pretrained_dict)

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
                len(images), len(self.model_paths[hemi]["anterior"]), self.num_voxels
            )
            for axis in ["anterior", "posterior"]:
                model_paths = self.model_paths[hemi][axis]

                for idx, model_path in enumerate(
                    tqdm(
                        model_paths,
                        desc=f"Running inference on {hemi} {axis} models",
                    )
                ):
                    preds = self.forward_region(model_path, images)
                    preds = torch.nan_to_num(preds)
                    hemi_preds[:, idx, :] += preds

            # coor_sm has shape (num_models, num_vertices)
            # corr_sm = (
            #     self.corr_sm[hemi]
            #     .unsqueeze(1)
            #     .expand(-1, hemi_preds.size(1), -1)
            #     .to("cpu")
            # )
            # print("corr_sm", corr_sm.shape)
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
        output, _, _ = evaluate(
            args, model, criterion, imgs_loader, imgs_dataset, print_freq=1000
        )

        return output
