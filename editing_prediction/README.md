# Training
We build an editing prediction model to predict whether a query video contains copied segments and what type of manipulations are used. We first create a simulated copy video dataset using the training reference of the VSC2022 data. Then we train a prediction model using this dataset.
## Generate simulated dataset
1. Download The training set of [VSC2022](https://www.drivendata.org/competitions/group/meta-video-similarity/).
2. Set the path of input video `data_path`, metadata `meta_path`, and output folder `output`. Run the following will generate samples with index from 0 to 4999.
```
python video_generation_AugLy.py --sample_range 0 5000
```
3. The generated data has the following components:
    - copied videos(query/Q4XXXXX.mp4)
    - ground-truth (cvl_train_matching_ground_truth.csv): query_id, ref_id, query_start/end, ref_start/end
    - metadata (cvl_train_query_metadata.csv): video_id duration_sec frames_per_sec width height rn base_query_id
    - manipulation type (cvl_train_augly_metadata/Q4XXXXX.json)
