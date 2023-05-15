# Meta-AI-Video-Similarity-Challenge-3rd-Place-Solution

The 3rd Place Solution of the Meta AI Video Similarity Challenge : Descriptor Track and Matching Track.

You can check our solution tech report from: [3rd Place Solution to Meta AI Video Similarity Challenge](https://arxiv.org/abs/2304.11964)

Please submit an issue rather than PR if you have any suggestions.

## Solution Overview

- Our solution is based on ISC-dt1 (https://github.com/lyakaap/ISC21-Descriptor-Track-1st), which is the 1st place solution of the Descriptor Track in the Image Similarity Challenge 2021.
- TN method for video temporal alignment: We use TN, which is proposed in the paper "Scalable detection of partial near-duplicate videos by visual-temporal consistency", to match the query and reference video pairs and to localize the copied frames in the Matching Track.
- Test time augmentation (TTA): We realized that TTA significantly boosts the score. The augmentation method used is selected from: split the image vertically (2 views), horizontally (2 views), or both (4 views).
- Editing prediction: We train a model that predicts what types of edits are used for the generation of query videos. Based on its predictions, we can select the suitable augmentation method for TTA. This helps to decrease the number of total query video descriptors. The training code for editing prediction is available in [editing_prediction/training](editing_prediction/training), and how to prepare the training data is described in [editing_prediction/README.md](editing_prediction/README.md).
- Descriptor Post-processing: Score Normalization is used to suppress the effect of bad matches (e.g., fully white or black frames occur in many videos, and these are matched with too high a score). We also concatenate adjacent frame descriptors to consider temporal information and apply PCA to reduce its dimension.
- Acceleration: The video encoding library, decord, is used to read video frames. We also use the GPU to accelerate input pre-processing, such as resizing and normalization. We tried to use the GPU for video loading, but it is difficult to use in the runtime environments without any issues.

<!-- - training: see [training/README.md](training/README.md) -->
<!-- - reproduce descriptor track: see [meta-vsc-descriptor-runtime/README.md](meta-vsc-descriptor-runtime/README.md) -->
<!-- - reproduce matching track: see [meta-vsc-matching-runtime/README.md](meta-vsc-matching-runtime/README.md) -->

## Reproduce our solution

You can reproduce our solution by following steps:

1. Finish data setup.
2. Train the editing prediction model. Instructions for reproducing the training of this model are described in [editing_prediction/README.md](editing_prediction/README.md).
3. Run the descriptor track runtime following this instruction: [meta-vsc-descriptor-runtime/README.md](meta-vsc-descriptor-runtime/README.md).
4. Run the matching track runtime following this instruction:: [meta-vsc-matching-runtime/README.md](meta-vsc-matching-runtime/README.md).

### Setup

Define environment variables (change the paths to your local paths):

```
export COMPETITION_DATA=/app/data/vsc/competition_data
```

Download the competition data:

```
aws s3 cp s3://drivendata-competition-meta-vsc-data-sg/train/ ${COMPETITION_DATA}/train/ --recursive --no-sign-request
aws s3 cp s3://drivendata-competition-meta-vsc-data-sg/test/ ${COMPETITION_DATA}/test/ --recursive --no-sign-request
aws s3 cp s3://drivendata-competition-meta-vsc-data-sg/phase_2_uB82/ ${COMPETITION_DATA}/phase_2_uB82/ --recursive --no-sign-request
```

## Reference
- https://github.com/drivendataorg/meta-vsc-descriptor-runtime
- https://github.com/drivendataorg/meta-vsc-matching-runtime
- https://github.com/facebookresearch/vsc2022
- https://github.com/lyakaap/ISC21-Descriptor-Track-1st
