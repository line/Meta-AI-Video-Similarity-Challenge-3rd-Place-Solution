# meta-vsc-matching-runtime

We will introduce how to reproduce our best results of the matching track phase-2.

## Setup

Prepare the data:

```
make update-submodules
make data-subset DATASET=train SUBSET_PROPORTION=0.1
make data-subset DATASET=test SUBSET_PROPORTION=0.1
```

Build docker image and enter the container:

```
make interact-container
```

Execute the following commands in the container:

```
cd submission_src

COMPETITION_DATA=/share-cvlab/yokoo/vsc/competition_data  # set the path to the competition data
OUTPUT_DIR=outputs  # set the path to the output directory

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
