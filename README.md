# whole_brain_encoder

# Required setup

In `utils/args.py`, modify the following paths in the default arguments:

1. `--output_path`: use the path to downloaded weights or folder where weights were saved after training
1. `--data_dir`: directory containing metadata and neural data files
1. `--imgs_dir`: directory containing NSD image data

## Environment setup

Environment for model training and inference

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

# Run inference on your images

Download the model weights and brain parcels from https://huggingface.co/ehwang/brain_encoder_weights/tree/main, or follow the training instructions below in [Reproducing the checkpoints](#reproducing-the-checkpoints).

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