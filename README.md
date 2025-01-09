# whole_brain_encoder

# Run inference on your images using the ensemble model

The ensemble model is a combination of many individual models, each trained on a particular subject, encoder layer, and hemisphere. It uses the validation split to generate a model confidence value for each voxel, so it should be evaluated on the test split or unseen data.

## Environment setup

### Downloading weights

In `utils/args.py`, set the default argument `--output_path` to point to downloaded weights or folder where weights were saved after training. Download the model weights from https://huggingface.co/ehwang/brain_encoder_weights/tree/main by cloning the repo like [this](https://huggingface.co/ehwang/brain_encoder_weights/tree/main?clone=true), or follow the training instructions below in [Reproducing the checkpoints](#reproducing-the-checkpoints). This should point to the `checkpoints/` folder.

The expected directory structure is: `checkpoints/nsd_test/dinov2_q_transformer/subj_{subj_num:02}/enc_{enc_layer}/run_{run_num}/{hemi}`.

### Conda environment for model training and inference

Environments for running the model, plotting, and parcellation algorithm are provided in the `env/` folder. To create the environments, run the following commands:

Environment for inference and training

```bash
conda env create -f env/xformers.yml
```

Environment for pycortex plotting

```bash
conda env create -f env/pycortex.yml
```

Environment for running the parcellation algorithm

```bash
conda env create -f env/parcel.yml
```

## Running inference

Follow the example in `tutorials/test_wrapper.ipynb`

# Plot ensemble results

```bash
python plot_run_results.py --ensemble 1 --subj $SUBJECT --split $SPLIT
```

valid splits include `train`, `val`, and `test`

# Plot results for a run

```bash
python plot_run_results.py --subj $SUBJECT --enc_output_layer $layer --run $RUN_ID
```

# Reproducing the checkpoints

## Required setup

In `utils/args.py`, modify the following paths in the default arguments:

1. `--data_dir`: Directory containing metadata and neural data files. These are from NSD.
1. `--imgs_dir`: Directory containing NSD image data. These are from NSD.

## Generating the brain parcels

We used a kmeans-based parcellation algorithm that generates around 450-500 parcels per hemisphere. To generate the brain parcels, run the following command:

```bash
conda activate parcel
python generate_parcels.py --subj $SUBJECT --hemi $HEMI --save_dir /path/to/save/
```

## Model training

If you're using slurm, see `scripts/train_plot` for a bash script to reproduce the checkpoint files.

Otherwise, to train a single model, run:

```bash
python main.py --subj $SUBJECT --enc_output_layer $layer --run $RUN_ID --hemi $HEMI
```

If you want to reproduce the checkpoints in the ensemble model, you want to train two runs each for
- `lh` and `rh` hemispheres
- encoder layers 1, 3, 5, 7 from the dino backbone