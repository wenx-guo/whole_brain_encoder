# whole_brain_encoder

# Run inference on your images using the ensemble model

The ensemble model is a combination of many individual models, each trained on a particular subject, encoder layer, and hemisphere. It uses the validation split to generate a model confidence value for each voxel, so it should be evaluated on the test split or unseen data.

## Environment setup

### Downloading weights

In `utils/args.py`, set the default argument `--output_path` to point to downloaded weights or folder where weights were saved after training. Download the model weights from https://huggingface.co/ehwang/brain_encoder_weights/tree/main by cloning the repo like [this](https://huggingface.co/ehwang/brain_encoder_weights/tree/main?clone=true), or follow the training instructions below in [Reproducing the checkpoints](#reproducing-the-checkpoints). This should point to the `checkpoints/` folder.

The expected directory structure is: `checkpoints/nsd_test/dinov2_q_transformer/subj_{subj_num:02}/enc_{enc_layer}/run_{run_num}/{hemi}`.

The hosted ensemble model includes weights for two runs each for
- `lh` and `rh` hemispheres
- encoder layers 1, 3, 5, 7 from the dino backbone

### Conda environment for model training and inference

Set up the conda environment for the model:

```bash
conda env create -f env/xformers.yml
```

## Running inference

Follow the example in `tutorials/test_wrapper.ipynb`

# Plotting results

Set up the conda environment for [pycortex](https://github.com/gallantlab/pycortex) plotting:

```bash
conda env create -f env/pycortex.yml
```

## Plot ensemble results

```bash
conda activate pycortex
python plot_run_results.py --ensemble 1 --subj $SUBJECT --split $SPLIT
```

The following directory structure is expected: `results/enc_{"_".join(enc_layers)}_run_{"_".join(run_nums)}/subj_{subj_num:02}/{hemi}_test_corr_avg.npy`, for both hemispheres lh and rh. For example, `results/enc_1_3_5_7_run_1_2/subj_01/lh_test_corr_avg.npy`. Plots for the correlation across all voxels and correlation for known ROIs will be saved in the same directory.

Valid splits include `train`, `val`, and `test`

## Plot results for a run

```bash
conda activate pycortex
python plot_run_results.py --subj $SUBJECT --enc_output_layer $layer --run $RUN_ID
```

The directory structure described in [Downloading weights](#downloading-weights) is expected. Plots for the correlation across all voxels and a graph showing correlation for known ROIs will be saved in the `run_{run_num}` directory.

# Reproducing the checkpoints

## Required setup

In `utils/args.py`, modify the following paths in the default arguments:

1. `--data_dir`: Directory containing metadata and neural data files. These are from NSD.
1. `--imgs_dir`: Directory containing NSD image data. These are from NSD.

## Generating the brain parcels

Set up the conda environment for the parcellation algorithm:

```bash
conda env create -f env/parcel.yml
```

We used a kmeans-based parcellation algorithm that generates around 450-500 parcels per hemisphere. To generate the brain parcels, run the following command:

```bash
conda activate parcel
python generate_parcels.py --subj $SUBJECT --hemi $HEMI --save_dir /path/to/save/
```

## Model training

Set up the [conda environment for model training and inference](#conda-environment-for-model-training-and-inference).

If you're using slurm, see `scripts/train_plot` for a bash script to reproduce the checkpoint files.

Otherwise, to train a single model, run:

```bash
conda activate xformers
python main.py --subj $SUBJECT --enc_output_layer $layer --run $RUN_ID --hemi $HEMI
```