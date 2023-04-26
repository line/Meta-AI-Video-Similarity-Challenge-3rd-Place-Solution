# meta-vsc-descriptor-runtime

We will introduce how to reproduce our best results of the descriptor track phase-2.

## Setup

Prepare the data:

```
make update-submodules
make data-subset DATASET=train SUBSET_PROPORTION=0.1
make data-subset DATASET=test SUBSET_PROPORTION=0.1
```

Prapre the model weights:

```
wget https://github.com/lyakaap/ISC21-Descriptor-Track-1st/releases/download/v1.0.1/isc_ft_v107.pth.tar -P submission_src/model_assets
wget https://github.com/lyakaap/ISC21-Descriptor-Track-1st/releases/download/v1.0.3/disc21_ft_vit_base_r50_s16_224_in21k.pth  -P submission_src/model_assets
cp your/trained/editing_prediction/model.ckpt submission_src/model_assets/copy_type_pred__convnext_clip.ckpt
```

Set environment variables:

```
COMPETITION_DATA=/share-cvlab/yokoo/vsc/competition_data  # set the path to the competition data
OUTPUT_DIR=outputs  # set the path to the output directory
```

Build docker image and enter the container:

```
make interact-container COMPETITION_DATA=${COMPETITION_DATA}
```

Execute the following commands in the container:

```
cd submission_src

# extract features
conda run --no-capture-output -n condaenv python -m src.inference_full \
    --accelerator=cuda --processes=8 --read_type tensor --video_reader DECORD \
    --dataset_paths ${COMPETITION_DATA}/phase_2_uB82/query ${COMPETITION_DATA}/test/reference ${COMPETITION_DATA}/train/reference \
    --output_path ${OUTPUT_DIR} --gt_path ${COMPETITION_DATA}/train/train_matching_ground_truth.csv --tta --fps 2 --stride 3

# create files for submission
conda run --no-capture-output -n condaenv python -m src.inference_full --mode test \
    --accelerator=cuda --processes=8 --read_type tensor --video_reader DECORD \
    --dataset_paths ${COMPETITION_DATA}/phase_2_uB82/query ${COMPETITION_DATA}/test/reference ${COMPETITION_DATA}/train/reference \
    --output_path ${OUTPUT_DIR} --tta --fps 2 --stride 3
```

Finally, the submission files are created by the following commands:

```
sudo make copy-full-results FULL_RESULTS_DIR=${OUTPUT_DIR}
sudo make force-pack-submission
sudo make test-submission
```
