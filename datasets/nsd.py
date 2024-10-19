import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from nilearn import datasets
import h5py
from nilearn import plotting
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr as corr
import torch.nn.functional as F


class nsd_dataset(Dataset):
    def reformat_parcels(self, original_parcels, parcels, position=[]):
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
            self.reformat_parcels(
                original_parcels,
                parcels[torch.where(parcels[:, 0] == p)[0]][:, 1:],
                [p.item()],
            )
            for p in torch.unique(parcels[:, 0])
        ]

    def __init__(self, args, split="train", parcel_paths=None, transform=None):
        super(nsd_dataset, self).__init__()

        self.subj = args.subj
        self.transform = transform
        self.backbone_arch = args.backbone_arch

        neural_data_path = Path(
            "/engram/nklab/datasets/natural_scene_dataset/model_training_datasets/neural_data"
        )
        metadata = np.load(
            neural_data_path / f"metadata_sub-{self.subj:02}.npy", allow_pickle=True
        ).item()
        self.img_order = metadata["img_presentation_order"]

        assert split in [
            "train",
            "test",
            "val",
        ], "split must be either train, test or val"
        split_imgs = metadata[f"{split}_img_num"]
        self.split_idxs = np.where(
            np.isin(metadata["img_presentation_order"], split_imgs)
        )[0]

        self.betas = h5py.File(neural_data_path / f"betas_sub-{self.subj:02}.h5", "r")

        imgs_dir = Path(
            "/engram/nklab/datasets/natural_scene_dataset/nsddata_stimuli/stimuli/nsd"
        )
        self.imgs = h5py.File(imgs_dir / "nsd_stimuli.hdf5", "r")

        parcel_path = Path(
            "/engram/nklab/algonauts/ethan/parcelling/results/plot_data_driven_parcellations"
        )
        self.parcels = {
            "lh_betas": torch.from_numpy(
                np.load(parcel_path / "lh_1000_kmeans_5init_3000iter_labels_s01.npy")
            ),
            "rh_betas": torch.from_numpy(
                np.load(parcel_path / "rh_1000_kmeans_5init_3000iter_labels_s01.npy")
            ),
        }
        self.max_parcel_size = max(
            torch.max(
                torch.unique(
                    self.parcels[hemi], dim=0, return_inverse=True, return_counts=True
                )[2]
            )
            for hemi in self.parcels.keys()
        )
        for hemi in self.parcels.keys():
            self.parcels[hemi] = self.parcels[hemi][:, None]
            self.parcels[hemi] = torch.tensor(
                [[0 if x[0] < 500 else 1, x[0]] for x in self.parcels[hemi]]
            )

            self.parcels[hemi] = self.reformat_parcels(
                self.parcels[hemi], self.parcels[hemi]
            )

    def transform_img(self, img):
        img = Image.fromarray(img)
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
        if any(isinstance(element, torch.Tensor) for element in labels):
            fmri = []

            for parcel in labels:
                parcel_data = fmri_data[parcel]
                pad_size = self.max_parcel_size - parcel_data.size(0)
                fmri.append(F.pad(parcel_data, (0, pad_size), mode="constant", value=0))

            return torch.stack(fmri)

        return [self.parcellate_fmri(fmri_data, parcel) for parcel in labels]

    def __getitem__(self, idx):
        idx = self.split_idxs[idx]

        img_ind = self.img_order[idx]  # image index in nsd
        img = self.imgs["imgBrick"][img_ind]
        img = self.transform_img(img)

        fmri_data = {
            hemi: self.parcellate_fmri(
                torch.from_numpy(self.betas[hemi][idx]), self.parcels[hemi]
            )
            for hemi in self.betas.keys()
        }
        return img, fmri_data

    def __len__(self):
        return len(self.split_idxs)


class algonauts_dataset(Dataset):
    def __init__(self, args, is_train, imgs_paths, idxs, transform=None):
        super(algonauts_dataset, self).__init__()
        self.imgs_paths = np.array(imgs_paths)[idxs]
        self.transform = transform
        self.is_train = is_train
        self.saved_feats = args.saved_feats
        dino_feat_dir = args.saved_feats_dir + "/dinov2_q_last/" + args.subj
        clip_feat_dir = args.saved_feats_dir + "/clip_vit_512/" + args.subj

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


def make_coco_transforms():
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    return T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            normalize,
        ]
    )

    raise ValueError(f"unknown {image_set}")


def fetch_dataloaders(
    args, train="train", shuffle=True, train_val_split="none", download=True
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
    kwargs = (
        {"num_workers": 0, "pin_memory": False} if torch.cuda.is_available() else {}
    )

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

        if args.output_path:
            np.save(args.save_dir + "/idxs.npy", idxs)

        # Assign 90% of the shuffled stimulus images to the training partition,
        # and 10% to the test partition
        idxs_train, idxs_val = idxs[:num_train], idxs[num_train:]

        train_imgs_paths = sorted(list(Path(train_img_dir).iterdir()))

        # The DataLoaders contain the ImageDataset class
        train_dataloader = DataLoader(
            algonauts_dataset(
                args, train, train_imgs_paths, idxs_train, transform_train
            ),
            shuffle=shuffle,
            batch_size=args.batch_size,
        )
        val_dataloader = DataLoader(
            algonauts_dataset(args, train, train_imgs_paths, idxs_val, transform_val),
            batch_size=args.batch_size,
        )
        print("Training stimulus images: " + format(len(idxs_train)))
        print("Validation stimulus images: " + format(len(idxs_val)))
        return train_dataloader, val_dataloader

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


#     img_folder = '../data/svrt_dataset/a128_results_problem_1'   # svrt_task1_64x64' #a128_results_problem_1'
#     transforms = T.Compose([T.ToTensor()])

#     dataset_ = algonauts_dataset(args, is_train=train, transforms=make_coco_transforms())
#     dataloader = torch.utils.data.DataLoader(dataset=dataset_, batch_size=args.batch_size, shuffle=shuffle, num_workers=0)


#     return dataloader
