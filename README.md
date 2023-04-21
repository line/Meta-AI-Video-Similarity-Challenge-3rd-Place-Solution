# Meta-AI-Video-Similarity-Challenge-3rd-Place-Solution

The 3rd Place Solution of the Meta AI Video Similarity Challenge : Descriptor Track and Matching Track.

You can check our solution tech report from: (WIP)

Please submit an issue rather than PR if you have any suggestions.

## Solution Overview

- Our solution is based on ISC-dt1 (https://github.com/lyakaap/ISC21-Descriptor-Track-1st), which is the 1st place solution of the Descriptor Track in the Image Similarity Challenge 2021.
- In addition to ISC-dt1, a ViT model with ResNet-50 as its stem, trained on the DISC21 dataset, is used as a feature extractor. Instructions for reproducing the training of this model are described in [training/README.md](training/README.md).
  - This ViT model is only used in the Matching Track for ensemble. Ensemble is also effective in the Descriptor Track, but we failed to submit the result of this model due to a network issue.
  - In our experiments, video modeling is not successful, and image modeling is sufficient for the copy detection task. Therefore, we treat each frame independently for feature extraction.
- TN method for video temporal alignment: We use TN, which is proposed in the paper "Scalable detection of partial near-duplicate videos by visual-temporal consistency", to match the query and reference video pairs and to localize the copied frames in the Matching Track.
- Test time augmentation (TTA): We realized that TTA significantly boosts the score. The augmentation method used is selected from: split the image vertically (2 views), horizontally (2 views), or both (4 views).
- Edit prediction: We train a model that predicts what types of edits are used for the generation of query videos. Based on its predictions, we can select the suitable augmentation method for TTA. This helps to decrease the number of total query video descriptors.
- Utilize video metadata: We observed that when a video is inserted with a segment from another video, one of its metadata, the sample aspect ratio (SAR), changes from a missing value to some specific value. We assume this is a natural phenomenon when creating copied videos, and this information is useful for distinguishing copied videos from non-copied videos. We filter out videos with missing SAR values as non-copied videos.
- Descriptor Post-processing: Score Normalization is used to suppress the effect of bad matches (e.g., fully white or black frames occur in many videos, and these are matched with too high a score). We also concatenate adjacent frame descriptors to consider temporal information and apply PCA to reduce its dimension.
- Acceleration: The video encoding library, decord, is used to read video frames. We also use the GPU to accelerate input pre-processing, such as resizing and normalization. We tried to use the GPU for video loading, but it is difficult to use in the runtime environments without any issues.

<!-- - training: see [training/README.md](training/README.md) -->
<!-- - reproduce descriptor track: see [meta-vsc-descriptor-runtime/README.md](meta-vsc-descriptor-runtime/README.md) -->
<!-- - reproduce matching track: see [meta-vsc-matching-runtime/README.md](meta-vsc-matching-runtime/README.md) -->

## Reference
- https://github.com/drivendataorg/meta-vsc-descriptor-runtime
- https://github.com/drivendataorg/meta-vsc-matching-runtime
- https://github.com/facebookresearch/vsc2022
- https://github.com/lyakaap/ISC21-Descriptor-Track-1st
