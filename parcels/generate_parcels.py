import numpy as np
import os
from cuml import KMeans
import dask.array as da
from pathlib import Path
import time
import argparse


def generate_parcels(hemi, subj_num, method="kmeans"):
    file = f"{hemi}_betas_sub-0{subj_num}.npy"
    print("target:", file)

    nclusters = 1000
    print("nclusters:", nclusters)

    save_dir = Path("/engram/nklab/algonauts/ethan/parcelling/results/")
    print("save_dir:", save_dir)
    os.makedirs(save_dir / f"{nclusters}_{method}_5init_3000iter_train", exist_ok=True)

    data_dir = Path(
        "/engram/nklab/datasets/natural_scene_dataset/model_training_datasets/neural_data"
    )

    betas = np.load(os.path.join(data_dir, file))

    betas = betas.T
    print("num voxels * num images:", betas.shape)

    metadata = np.load(
        data_dir / f"metadata_sub-{subj_num:02}.npy", allow_pickle=True
    ).item()
    split_imgs = metadata["train_img_num"]
    split_idxs = np.where(np.isin(metadata["img_presentation_order"], split_imgs))[0]

    betas = betas[:, split_idxs]

    betas = betas.astype(np.float32)

    betas = da.from_array(betas, chunks=(8192, 30000))

    algo1 = KMeans(
        n_clusters=nclusters,
        verbose=True,
        output_type="numpy",
        n_init=5,
        max_iter=3000,
    )
    algo1.fit(betas)

    algo2 = KMeans(
        n_clusters=2,
        verbose=True,
        output_type="numpy",
        n_init=5,
        max_iter=3000,
    )
    algo2.fit(algo1.cluster_centers_)

    algo2_labels = algo2.labels_[algo1.labels_]

    labels = np.hstack((algo2_labels[:, None], algo1.labels_[:, None]))

    np.save(
        save_dir
        / f"{nclusters}_{method}_5init_3000iter_train"
        / f"{hemi}_labels_s{subj_num:02}.npy",
        labels,
    )


def cluster(data, num_clusters, n_init=5, max_iter=3000):
    data = data.astype(np.float32)

    da_data = da.from_array(data, chunks=(8192, 30000))

    algo1 = KMeans(
        n_clusters=num_clusters,
        verbose=True,
        output_type="numpy",
        n_init=n_init,
        max_iter=max_iter,
    )
    algo1.fit(da_data)

    return algo1.labels_, algo1.cluster_centers_


def adjust_cluster_sizes(
    data,
    labels,
    centers,
    min_size,
    max_size,
    n_init,
    max_iter,
    break_size=2,
):
    """
    Args:
        data : should be masked
        labels
        centers
        min_size
        max_size
    """

    uq = np.unique(labels, return_counts=True)
    it = 0
    while (max(np.unique(labels, return_counts=True)[1]) > max_size) or (
        (np.unique(labels, return_counts=True)[1] < min_size).sum() > 1
    ):
        uq = np.unique(labels, return_counts=True)
        print(
            f"[{it}] num_clusters: {len(uq[1])}, num_large_offending: {np.sum(uq[1] > max_size)}, num_small_offending: {np.sum(uq[1] < min_size)}, max_size: {max(uq[1])}, min_size: {min(uq[1])}"
        )
        it += 1

        # while max(np.unique(labels, return_counts=True)[1]) > max_size:
        if max(np.unique(labels, return_counts=True)[1]) > max_size:
            uq = np.unique(labels, return_counts=True)

            largest_cluster = uq[0][np.argmax(uq[1])]

            largest_cluster_idx = np.where(labels == largest_cluster)[0]

            cluster_data = data[largest_cluster_idx]

            num_chunks = np.random.randint(2, break_size + 1)

            new_labels, new_centers = cluster(
                cluster_data,
                num_chunks,
                n_init=n_init,
                max_iter=max_iter,
            )

            new_labels = new_labels + np.max(labels) + 1

            labels[largest_cluster_idx] = new_labels
            centers = np.vstack((centers, new_centers))

        if (np.unique(labels, return_counts=True)[1] < min_size).sum() > 1:
            uq = np.unique(labels, return_counts=True)

            clusters_below_min_size = uq[0][np.where(uq[1] < min_size)[0]]

            centers_below_min_size = centers[clusters_below_min_size]

            # using corr
            # centers_corr = np.abs(np.corrcoef(centers_below_min_size))
            # np.fill_diagonal(centers_corr, np.nan)

            # using euclidean distance
            diff = (
                centers_below_min_size[:, np.newaxis, :]
                - centers_below_min_size[np.newaxis, :, :]
            )
            centers_corr = np.sum(diff**2, axis=-1)
            np.fill_diagonal(centers_corr, np.nan)

            pair_to_fuse = np.array(
                np.unravel_index(np.nanargmax(centers_corr), centers_corr.shape)
            )
            pair_to_fuse = clusters_below_min_size[pair_to_fuse]
            labels[np.where(labels == pair_to_fuse[1])[0]] = pair_to_fuse[0]

            centers[pair_to_fuse[0]] = np.mean(
                data[np.where(labels == pair_to_fuse[0])[0]], axis=0
            )

    return labels, centers


def ft_parcellation_algo(subj, hemi, n_init, max_iter):
    neural_data_path = Path(
        "/engram/nklab/datasets/natural_scene_dataset/model_training_datasets/neural_data"
    )
    metadata = np.load(
        neural_data_path / f"metadata_sub-{subj:02}.npy", allow_pickle=True
    ).item()

    data_dir = Path(
        "/engram/nklab/datasets/natural_scene_dataset/model_training_datasets/neural_data"
    )

    file = f"{hemi}_betas_sub-0{subj}.npy"
    betas = np.load(os.path.join(data_dir, file))
    betas = betas.T
    print("num voxels * num images:", betas.shape)

    nclusters = 200

    cluster_labels = np.zeros((betas.shape[0], 2))
    for i, axis in enumerate(["anterior", "posterior"]):
        mask = metadata[f"{hemi}_{axis}_vertices"]

        masked_betas = betas[mask]

        labels, centers = cluster(
            masked_betas,
            nclusters,
            n_init=n_init,
            max_iter=max_iter,
        )

        # use 10th and 90th percentiles of cluster sizes as min and max
        counts = np.sort(np.unique(labels, return_counts=True)[1])
        min_cluster_size = counts[len(counts) // 10 * 2]
        max_cluster_size = counts[len(counts) // 10 * 8]

        adjusted_labels, adjusted_centers = adjust_cluster_sizes(
            masked_betas,
            labels.copy(),
            centers.copy(),
            min_cluster_size,
            max_cluster_size,
            n_init=n_init,
            max_iter=max_iter,
            break_size=4,
        )

        print(labels.shape, mask.shape)
        print(
            np.stack([np.full_like(adjusted_labels, i), adjusted_labels], axis=1).shape
        )

        cluster_labels[mask] = np.stack(
            [np.full_like(adjusted_labels, i), adjusted_labels], axis=1
        )

    return cluster_labels


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--subj", type=int, default=1)
    args.add_argument("--hemi", type=str, default="lh")
    args.add_argument(
        "--save_dir",
        type=str,
        default="/engram/nklab/algonauts/ethan/parcelling/results/200c_20percentile_ftalgo_5init_3000iter_train",
    )
    args = args.parse_args()


    for subj in range(1, 9):
        for hemi in ["lh", "rh"]:
            save_dir = Path(args.save_dir)
            os.makedirs(save_dir, exist_ok=True)

            cluster_labels = ft_parcellation_algo(
                subj, hemi, n_init=5, max_iter=3000
            )

            np.save(
                save_dir / f"{hemi}_labels_s{subj:02}.npy",
                cluster_labels,
            )


if __name__ == "__main__":
    main()
