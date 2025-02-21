import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
import h5py
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
from itertools import chain


class nsd_dataset_tempate(Dataset):
    def __init__(self, args, split="train", transform=None):
        self.subj = int(args.subj)
        self.hemi = args.hemi
        self.transform = transform
        self.backbone_arch = args.backbone_arch

        neural_data_path = Path(args.data_dir)
        self.metadata = np.load(
            neural_data_path / f"metadata_sub-{self.subj:02}.npy", allow_pickle=True
        ).item()
        self.img_order = self.metadata["img_presentation_order"]

        assert split in [
            "train",
            "test",
            "val",
        ], "split must be either train, test, val, or custom"
        self.split_imgs = self.metadata[f"{split}_img_num"]

        if self.hemi is not None:
            self.betas = h5py.File(
                neural_data_path / f"betas_sub-{self.subj:02}.h5", "r"
            )[f"{self.hemi}_betas"]
        else:
            self.betas = [
                h5py.File(neural_data_path / f"betas_sub-{self.subj:02}.h5", "r")[
                    f"{hemi}_betas"
                ]
                for hemi in ["lh", "rh"]
            ]

        imgs_dir = Path(args.imgs_dir)
        self.imgs = h5py.File(imgs_dir / "nsd_stimuli.hdf5", "r")

        parcel_path = Path(args.parcel_dir)
        if args.hemi is not None:
            self.parcels = torch.load(
                parcel_path / f"{args.hemi}_labels_s{self.subj:02}.pt",
                weights_only=True,
            )
        else:
            self.parcels = torch.load(
                parcel_path / f"labels_s{self.subj:02}.pt", weights_only=True
            )
        # we are only interested in evaluating on these voxels
        parcel_idxs = torch.cat([p.flatten() for p in self.parcels], dim=0)
        parcel_idxs = torch.unique(parcel_idxs)
        if self.hemi is not None:
            self.valid_voxel_mask = torch.zeros(len(self.betas[0]), dtype=torch.bool)
        else:
            self.valid_voxel_mask = torch.zeros(
                sum([len(b[0]) for b in self.betas]), dtype=torch.bool
            )
        for parcel in self.parcels:
            self.valid_voxel_mask[parcel] = True
        self.num_hemi_voxels = torch.sum(self.valid_voxel_mask).item()
        print("Number of valid voxels: ", self.num_hemi_voxels)

        self.num_parcels = len(self.parcels)
        print("Number of parcels: ", self.num_parcels)

    def plot_parcels(self):
        if self.overlap:
            print("Cannot plot overlapping parcels")
            return

        import cortex
        import cortex.polyutils
        import contextlib
        from io import StringIO
        import sys

        @contextlib.contextmanager
        def suppress_print():
            original_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                yield
            finally:
                sys.stdout = original_stdout

        def plot_parcels(
            lh, rh, title="", fig_path=None, cmap="freesurfer_aseg_256", clip=1
        ):
            plt.rc("xtick", labelsize=19)
            plt.rc("ytick", labelsize=19)

            subject = "fsaverage"
            data = np.append(lh, rh)
            vertex_data = cortex.Vertex(
                data, subject, cmap=cmap, vmin=0, vmax=clip
            )  # "afmhot"

            with suppress_print():
                cortex.quickshow(vertex_data, with_curvature=True)

            plt.title(title)

            if fig_path is not None:
                plt.savefig(fig_path, dpi=300)
            else:
                plt.show()

        fsavg = np.empty((max([torch.max(p) for p in self.parcels]) + 1))
        fsavg[:] = np.nan

        for idx, parcel in enumerate(self.parcels):
            fsavg[parcel.numpy()] = idx

        plot_parcels(
            fsavg if self.hemi == "lh" else np.full_like(fsavg, np.nan),
            fsavg if self.hemi == "rh" else np.full_like(fsavg, np.nan),
            clip=np.nanmax(fsavg),
        )

    def reformat_parcels(self, parcels, metaparcel_idx):
        """
        args:
        parcels: [[(level1, level2, ...), ...], [(level1, level2, ...), ...], ...]

        returns: [level1: [idx1, idx2, ...], level2: [idx1, idx2, ...], ...]
        """
        flattened_parcels = np.array(list(chain.from_iterable(parcels)))
        print(flattened_parcels)
        flattened_parcels = torch.from_numpy(flattened_parcels)
        flattened_parcels = flattened_parcels[flattened_parcels[:, 0] == metaparcel_idx]
        flattened_parcels = flattened_parcels[:, 1]
        uq_parcels = torch.unique(flattened_parcels)

        labels = [[] for _ in range(len(uq_parcels))]
        parcel_to_idx = {p.item(): i for i, p in enumerate(uq_parcels)}
        for v in range(len(parcels)):
            for affiliation in parcels[v]:
                if affiliation[0] != metaparcel_idx:
                    continue
                parcel_idx = parcel_to_idx[affiliation[1]]
                labels[parcel_idx].append(v)

        for i in range(len(labels)):
            labels[i] = torch.tensor(labels[i])

        return labels

    def reformat_parcels_nonoverlapping(self, original_parcels, parcels, position=[]):
        """
        args:
        parcels: [(level1, level2, ...), (level1, level2, ...), ...]

        returns: [level1: [idx1, idx2, ...], level2: [idx1, idx2, ...], ...]
        """
        if len(parcels[0]) == 1:
            t = [
                (original_parcels == torch.tensor(position + [p]))
                .all(dim=1)
                .nonzero(as_tuple=True)[0]
                for p in torch.unique(parcels)
            ]
            return t

        return [
            self.reformat_parcels_nonoverlapping(
                original_parcels,
                parcels[torch.where(parcels[:, 0] == p)[0]][:, 1:],
                [p.item()],
            )
            for p in torch.unique(parcels[:, 0])
        ]

    def transform_img(self, img):
        # img = Image.fromarray(img)
        # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')

        if self.transform:
            img = self.transform(img)

        if self.backbone_arch:
            if "dinov2" in self.backbone_arch:
                patch_size = 14

                size_im = (
                    img.shape[0],
                    int(np.ceil(img.shape[1] / patch_size) * patch_size),
                    int(np.ceil(img.shape[2] / patch_size) * patch_size),
                )
                paded = torch.zeros(size_im)
                paded[:, : img.shape[1], : img.shape[2]] = img
                img = paded

        return img

    def parcellate_fmri(self, fmri_data, labels):
        fmri = []
        for parcel in labels:
            parcel_data = fmri_data[parcel]
            pad_size = self.max_parcel_size - parcel_data.size(0)
            fmri.append(F.pad(parcel_data, (0, pad_size), mode="constant", value=0))
        return torch.stack(fmri)

    def get_parcel_mask(self):
        mask = torch.zeros(self.num_parcels, self.max_parcel_size, dtype=torch.bool)

        for i, parcel in enumerate(self.parcels):
            pad_size = self.max_parcel_size - parcel.size(0)
            if pad_size == 0:
                mask[i] = 1
            else:
                mask[i][:-pad_size] = 1

        return mask


class nsd_dataset(nsd_dataset_tempate):
    def __init__(
        self, args, split="train", parcel_path=None, transform=None, preload_data=False
    ):
        super().__init__(args, split, transform)

        self.split_idxs = np.where(
            np.isin(self.metadata["img_presentation_order"], self.split_imgs)
        )[0]

    def __getitem__(self, idx):
        split_idx = self.split_idxs[idx]

        img_ind = self.img_order[split_idx]  # image index in nsd
        img = self.imgs["imgBrick"][img_ind]
        img = self.transform_img(img)

        fmri_data = {}
        if self.hemi is not None:
            fmri_data["betas"] = torch.from_numpy(self.betas[split_idx])
        else:
            fmri_data["betas"] = torch.from_numpy(
                np.concatenate([b[split_idx] for b in self.betas])
            )

        return img, fmri_data

    def __len__(self):
        return len(self.split_idxs)


class nsd_dataset_avg(nsd_dataset_tempate):
    def __init__(
        self, args, split="train", parcel_paths=None, transform=None, preload_data=False
    ):
        super().__init__(args, split, transform)

        assert split in [
            "train",
            "test",
            "val",
        ], "split must be either train, test or val"

        # some of the images in split_imgs are were not actually presented, so let's take them out
        self.split_presented_imgs = self.split_imgs[
            np.isin(self.split_imgs, self.metadata["img_presentation_order"])
        ]
        self.img_to_runs = [
            np.where(self.metadata["img_presentation_order"] == img_ind)[0]
            for img_ind in self.split_presented_imgs
        ]

    def __getitem__(self, i):
        img_ind = self.split_presented_imgs[i]  # image index in nsd
        img = self.imgs["imgBrick"][img_ind]

        if self.transform is not None:
            img = self.transform_img(img)

        fmri_data = {}
        data_idxs = self.img_to_runs[i]

        if self.hemi is not None:
            data = torch.from_numpy(self.betas[data_idxs])
            data = torch.mean(data, axis=0)
            fmri_data["betas"] = data
        else:
            data = np.concatenate([b[data_idxs] for b in self.betas], axis=1)
            data = torch.from_numpy(data)
            data = torch.mean(data, axis=0)
            fmri_data["betas"] = data

        return img, fmri_data

    def __len__(self):
        return len(self.split_presented_imgs)


class nsd_dataset_custom(nsd_dataset_tempate):
    """For when you bring your own data"""

    def __init__(
        self,
        img_data,
        transform=None,
    ):
        self.transform = transform
        self.backbone_arch = "dinov2_q"

        self.img_data = img_data

    def __getitem__(self, idx):
        img = self.img_data[idx]
        img = self.transform_img(img)

        return img, {"betas": torch.empty((163842))}

    def __len__(self):
        return len(self.img_data)


class algonauts_dataset(Dataset):
    def __init__(
        self,
        args,
        is_train,
        imgs_paths,
        idxs,
        parcel_path,
        overlap,
        transform=None,
    ):
        super(algonauts_dataset, self).__init__()
        self.imgs_paths = np.array(imgs_paths)[idxs]
        self.transform = transform
        self.is_train = is_train
        self.saved_feats = args.saved_feats
        self.subj = int(args.subj)
        dino_feat_dir = (
            args.saved_feats_dir + "/dinov2_q_last/" + f"{int(args.subj):02}"
        )
        clip_feat_dir = args.saved_feats_dir + "/clip_vit_512/" + f"{int(args.subj):02}"

        self.backbone_arch = args.backbone_arch

        self.cat_clip = 1

        if is_train == "train":
            if self.saved_feats:
                fts_subj_train = np.load(dino_feat_dir + "/train.npy")
                clip_subj_train = np.load(clip_feat_dir + "/train.npy")
                self.fts_subj_train = fts_subj_train[idxs]
                self.clip_subj_train = clip_subj_train[idxs]

            fmri_dir = os.path.join(args.data_dir, "training_split", "training_fmri")
            lh_fmri = np.load(os.path.join(fmri_dir, "lh_training_fmri.npy"))
            rh_fmri = np.load(os.path.join(fmri_dir, "rh_training_fmri.npy"))
            self.lh_fmri = lh_fmri[idxs]
            self.rh_fmri = rh_fmri[idxs]

        elif is_train == "test":
            if self.saved_feats:
                self.fts_subj_test = np.load(dino_feat_dir + "/synt.npy")
                self.clip_subj_test = np.load(clip_feat_dir + "/synt.npy")

        self.length = len(idxs)

        self.hemi = args.hemi
        parcel_path = Path(parcel_path)
        self.axis_mask = None

        try:
            self.parcels = np.load(
                parcel_path / f"{args.hemi}_labels_s{self.subj:02}.npy"
            )
        except:
            self.parcels = np.load(
                parcel_path / f"{args.hemi}_labels_s{self.subj:02}.npy",
                allow_pickle=True,
            )

        self.num_hemi_voxels = len(self.parcels)

        if not overlap:
            self.parcels = torch.from_numpy(self.parcels)
            self.parcels = self.reformat_parcels_nonoverlapping(
                self.parcels, self.parcels
            )[args.metaparcel_idx]
        else:
            self.parcels = self.reformat_parcels(self.parcels, args.metaparcel_idx)

        self.max_parcel_size = max([len(p) for p in self.parcels])
        self.padded_parcels = torch.nn.utils.rnn.pad_sequence(
            self.parcels, batch_first=True, padding_value=-1
        )
        self.masks = self.padded_parcels != -1

        self.num_parcels = len(self.parcels)
        print("Number of parcels: ", self.num_parcels)

    def __getitem__(self, idx):
        if self.is_train == "train":
            if self.saved_feats:
                img = torch.tensor(self.fts_subj_train[idx])
                img = torch.reshape(img, (962, 768))

                if self.cat_clip:
                    clip_fts = torch.tensor(self.clip_subj_train[idx])
                    clip_fts = torch.tile(clip_fts[None, :], (img.shape[0], 1))
                    img = torch.cat((img, clip_fts), dim=1)
                    img = torch.reshape(img[1:, :], (31, 31, 512 + 768)).permute(
                        2, 0, 1
                    )

                    if self.saved_feats == "clip":
                        img = clip_fts
                        img = torch.reshape(img[1:, :], (31, 31, 512)).permute(2, 0, 1)

                else:
                    img = torch.reshape(img[1:, :], (31, 31, 768)).permute(2, 0, 1)

            else:
                img_path = self.imgs_paths[idx]
                img = Image.open(img_path).convert("RGB")
                # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')

                if self.transform:
                    img = self.transform(img)

                if self.backbone_arch:
                    if "dinov2" in self.backbone_arch:
                        patch_size = 14

                        size_im = (
                            img.shape[0],
                            int(np.ceil(img.shape[1] / patch_size) * patch_size),
                            int(np.ceil(img.shape[2] / patch_size) * patch_size),
                        )
                        paded = torch.zeros(size_im)
                        paded[:, : img.shape[1], : img.shape[2]] = img
                        img = paded

            lh_ = self.lh_fmri[idx]
            rh_ = self.rh_fmri[idx]

            fmri_data = {}
            fmri_data["lh_f"] = [lh_]
            fmri_data["rh_f"] = [rh_]

            fmri_data["betas"] = fmri_data[f"{self.hemi}_f"][0]

            return img, fmri_data  # lh_, rh_

        elif self.is_train == "test":
            if self.saved_feats:
                img = torch.tensor(self.fts_subj_test[idx])

                img = torch.reshape(img, (962, 768))

                if self.cat_clip:
                    clip_fts = torch.tensor(self.clip_subj_test[idx])
                    clip_fts = torch.tile(clip_fts[None, :], (img.shape[0], 1))
                    img = torch.cat((img, clip_fts), dim=1)

                    img = torch.reshape(img[1:, :], (31, 31, 512 + 768)).permute(
                        2, 0, 1
                    )

                    if self.saved_feats == "clip":
                        img = clip_fts
                        img = torch.reshape(img[1:, :], (31, 31, 512)).permute(2, 0, 1)

                else:
                    img = torch.reshape(img[1:, :], (31, 31, 768)).permute(2, 0, 1)

            else:
                img_path = self.imgs_paths[idx]
                img = Image.open(img_path).convert("RGB")
                # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')
                if self.transform:
                    img = self.transform(img)

                if self.backbone_arch:
                    if "dinov2" in self.backbone_arch:
                        patch_size = 14

                        size_im = (
                            img.shape[0],
                            int(np.ceil(img.shape[1] / patch_size) * patch_size),
                            int(np.ceil(img.shape[2] / patch_size) * patch_size),
                        )
                        paded = torch.zeros(size_im)
                        paded[:, : img.shape[1], : img.shape[2]] = img
                        img = paded

            return img

    def __len__(self):
        return self.length

    def reformat_parcels(self, parcels, metaparcel_idx):
        """
        args:
        parcels: [[(level1, level2, ...), ...], [(level1, level2, ...), ...], ...]

        returns: [level1: [idx1, idx2, ...], level2: [idx1, idx2, ...], ...]
        """
        flattened_parcels = np.array(list(chain.from_iterable(parcels)))
        flattened_parcels = torch.from_numpy(flattened_parcels)
        flattened_parcels = flattened_parcels[flattened_parcels[:, 0] == metaparcel_idx]
        flattened_parcels = flattened_parcels[:, 1]
        uq_parcels = torch.unique(flattened_parcels)

        labels = [[] for _ in range(len(uq_parcels))]
        parcel_to_idx = {p.item(): i for i, p in enumerate(uq_parcels)}
        for v in range(len(parcels)):
            for affiliation in parcels[v]:
                if affiliation[0] != metaparcel_idx:
                    continue
                parcel_idx = parcel_to_idx[affiliation[1]]
                labels[parcel_idx].append(v)

        for i in range(len(labels)):
            labels[i] = torch.tensor(labels[i])

        return labels

    def reformat_parcels_nonoverlapping(self, original_parcels, parcels, position=[]):
        """
        args:
        parcels: [(level1, level2, ...), (level1, level2, ...), ...]

        returns: [level1: [idx1, idx2, ...], level2: [idx1, idx2, ...], ...]
        """
        if len(parcels[0]) == 1:
            t = [
                (original_parcels == torch.tensor(position + [p]))
                .all(dim=1)
                .nonzero(as_tuple=True)[0]
                for p in torch.unique(parcels)
            ]
            return t

        return [
            self.reformat_parcels_nonoverlapping(
                original_parcels,
                parcels[torch.where(parcels[:, 0] == p)[0]][:, 1:],
                [p.item()],
            )
            for p in torch.unique(parcels[:, 0])
        ]

    def plot_parcels(self, cmap="cubehelix"):
        import cortex
        import cortex.polyutils
        import contextlib
        from io import StringIO
        import sys

        @contextlib.contextmanager
        def suppress_print():
            # Redirect stdout to suppress printing
            original_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                yield
            finally:
                # Restore original stdout after suppression
                sys.stdout = original_stdout

        def plot_parcels(
            lh, rh, title="", fig_path=None, cmap="freesurfer_aseg_256", clip=1
        ):
            plt.rc("xtick", labelsize=19)
            plt.rc("ytick", labelsize=19)

            subject = "fsaverage"
            data = np.append(lh, rh)
            vertex_data = cortex.Vertex(
                data, subject, cmap=cmap, vmin=0, vmax=clip
            )  # "afmhot"

            with suppress_print():
                cortex.quickshow(vertex_data, with_curvature=True)

            plt.title(title)

            if fig_path is not None:
                plt.savefig(fig_path, dpi=300)
            else:
                plt.show()

        challenge_data_dir = "/engram/nklab/algonauts/algonauts_2023_challenge_data/"

        mask_dir = os.path.join(
            challenge_data_dir,
            "subj" + format(self.subj, "02"),
            "roi_masks",
            f"{self.hemi}.all-vertices_fsaverage_space.npy",
        )
        fsaverage_all_vertices = np.load(mask_dir)
        fsavg = np.empty((len(fsaverage_all_vertices)))
        fsavg[:] = np.nan
        chall_area = np.zeros_like(np.where(fsaverage_all_vertices)[0])

        for idx, parcel in enumerate(self.parcels):
            chall_area[parcel.numpy()] = idx
        fsavg[np.where(fsaverage_all_vertices)[0]] = chall_area

        plot_parcels(
            fsavg if self.hemi == "lh" else np.full_like(fsavg, np.nan),
            fsavg if self.hemi == "rh" else np.full_like(fsavg, np.nan),
            clip=np.nanmax(fsavg),
            cmap=cmap,
        )


def fetch_dataloaders(
    args,
    overlap,
    parcel_path="/engram/nklab/algonauts/ethan/parcelling/results/algonauts_rois_all",
    train="train",
    shuffle=True,
    train_val_split="none",
    download=True,
):
    """
    load dataset depending on the task
    currently implemented tasks:
        -svhn
        -cifar10
        -mnist
        -multimnist, multimnist_cluttered
    args
        -args
        -batch size
        -train: if True, load train dataset, else test dataset
        -train_val_split:
            'none', load entire train dataset
            'train', load first 90% as train dataset
            'val', load last 10% as val dataset
            'train-val', load 90% train, 10% val dataset
    """

    transform_train = transforms.Compose(
        [
            #         transforms.RandomRotation(degrees=(0, 15)),
            #         transforms.RandomCrop(375),
            #         transforms.Resize((225,225)), # resize the images to 224x24 pixels
            transforms.ToTensor(),  # convert the images to a PyTorch tensor
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),  # normalize the images color channels
        ]
    )

    transform_val = transforms.Compose(
        [
            #         transforms.RandomCrop(400),
            #         transforms.Resize((225,225)), # resize the images to 224x24 pixels
            transforms.ToTensor(),  # convert the images to a PyTorch tensor
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),  # normalize the images color channels
        ]
    )
    args.data_dir = (
        Path("/engram/nklab/algonauts/algonauts_2023_challenge_data")
        / f"subj{int(args.subj):02}"
    )

    if train == "train":
        train_img_dir = os.path.join(args.data_dir, "training_split", "training_images")

        # Create lists will all training and test image file names, sorted
        train_img_list = os.listdir(train_img_dir)
        train_img_list = [f for f in train_img_list if f.endswith(".png")]
        train_img_list.sort()

        # rand_seed = 5 #@param
        # np.random.seed(rand_seed)

        # Calculate how many stimulus images correspond to 90% of the training data
        num_train = int(np.round(len(train_img_list) / 100 * 90))
        # Shuffle all training stimulus images
        idxs = np.arange(len(train_img_list))

        if args.run < 20:
            np.random.shuffle(idxs)

        # if args.output_path:
        #     np.save(args.save_dir + "/idxs.npy", idxs)

        # Assign 90% of the shuffled stimulus images to the training partition,
        # and 10% to the test partition
        idxs_train, idxs_val = idxs[:num_train], idxs[num_train:]

        train_imgs_paths = sorted(list(Path(train_img_dir).iterdir()))

        # The DataLoaders contain the ImageDataset class
        train_dataset = algonauts_dataset(
            args,
            train,
            train_imgs_paths,
            idxs_train,
            parcel_path,
            overlap,
            transform_train,
        )

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=shuffle,
            batch_size=args.batch_size,
        )
        val_dataloader = DataLoader(
            algonauts_dataset(
                args,
                train,
                train_imgs_paths,
                idxs_val,
                parcel_path,
                overlap,
                transform_val,
            ),
            batch_size=args.batch_size,
        )
        print("Training stimulus images: " + format(len(idxs_train)))
        print("Validation stimulus images: " + format(len(idxs_val)))
        return train_dataloader, val_dataloader, train_dataset

    elif train == "test":
        # test_img_dir  = os.path.join(args.data_dir, 'test_split', 'test_images')

        test_img_dir = os.path.join(args.data_dir, "../nsdsynthetic_stimuli/")

        test_img_list = os.listdir(test_img_dir)
        test_img_list.sort()

        test_imgs_paths = sorted(list(Path(test_img_dir).iterdir()))
        # No need to shuffle or split the test stimulus images
        idxs_test = np.arange(len(test_img_list))

        test_dataloader = DataLoader(
            algonauts_dataset(args, train, test_imgs_paths, idxs_test, transform_val),
            batch_size=args.batch_size,
        )
        print("\nTest stimulus images: " + format(len(idxs_test)))
        return test_dataloader
