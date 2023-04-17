# Training

## Setup

Setup submodules:

```
git submodule update --init --recursive
```

Define environment variables (change the paths to your local paths):

```
export COMPETITION_DATA=/app/data/vsc/competition_data
export DISC_DATA=/app/data/isc
```

Download the required data:

```
aws s3 cp s3://drivendata-competition-meta-vsc-data-sg/train/ ${COMPETITION_DATA}/train/ --recursive --no-sign-request
aws s3 cp s3://drivendata-competition-meta-vsc-data-sg/test/ ${COMPETITION_DATA}/test/ --recursive --no-sign-request
aws s3 cp s3://drivendata-competition-meta-vsc-data-sg/phase_2_uB82/ ${COMPETITION_DATA}/phase_2_uB82/ --recursive --no-sign-request
DISC_DATA=${DISC_DATA} bash download_DISC21.sh  # download DISC21 data
```

Setup data and docker environment:

```
docker build -t vsc-dev .

docker run -it --rm --gpus all --shm-size 256gb \
    -v `pwd`:/app \
    -v /_vsfs_managed/5b94c406-d5c4-430f-bb1e-15b3c0d138d7/share/yokoo/:/share-cvlab/yokoo/ \
    -v /_vsfs_managed/d61c6d35-f944-4908-9673-5b59244e1506/share/:/yokoo-data/ \
    -e COMPETITION_DATA=${COMPETITION_DATA} \
    -e DISC_DATA=${DISC_DATA} \
    vsc-dev bash
```

## Run training

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python v41.py \
    -a vit_base_r50_s16_224_in21k --seed 22999 --workers 8 \
    --epochs 10 --lr 0.001 --wd 1e-6 --optimizer sgd --batch-size 32 --val-batch-size 16 \
    --pos-margin 0.0 --neg-margin 1.0 --input-size 224 --memory-size 4096 --feature_dim 512 --sample_size 2000000 \
    --ddp_strategy deepspeed_stage_2 --warmup_steps 0 --num_frames 32 --eval_subset \
    --query_video_dir ${COMPETITION_DATA}/train/query \
    --ref_video_dir ${COMPETITION_DATA}/train/reference \
    --noise_video_dir ${COMPETITION_DATA}/test/reference \
    --eval_gt_path ${COMPETITION_DATA}/train/train_matching_ground_truth.csv \
    --query_metadata_path ${COMPETITION_DATA}/train/train_query_metadata.csv \
    --ref_metadata_path ${COMPETITION_DATA}/train/train_reference_metadata.csv \
    --noise_metadata_path ${COMPETITION_DATA}/test/test_reference_metadata.csv \
    ${DISC_DATA}

WEIGHT_PATH=v41/train_0314_024117/model.pth  # set the path to the trained model
python v44.py \
    -a vit_base_r50_s16_224_in21k --seed 722999 --workers 8 \
    --epochs 5 --lr 0.0005 --wd 1e-6 --optimizer sgd --batch-size 16 --val-batch-size 16 \
    --pos-margin 0.1 --neg-margin 1.1 --input-size 384 --memory-size 8192  --sample_size 2000000 --feature_dim 512 \
    --ddp_strategy deepspeed_stage_2 --warmup_steps 0 --num_frames 32 --eval_subset \
    --query_video_dir ${COMPETITION_DATA}/train/query \
    --ref_video_dir ${COMPETITION_DATA}/train/reference \
    --noise_video_dir ${COMPETITION_DATA}/test/reference \
    --eval_gt_path ${COMPETITION_DATA}/train/train_matching_ground_truth.csv \
    --query_metadata_path ${COMPETITION_DATA}/train/train_query_metadata.csv \
    --ref_metadata_path ${COMPETITION_DATA}/train/train_reference_metadata.csv \
    --noise_metadata_path ${COMPETITION_DATA}/test/test_reference_metadata.csv \
    --weight ${WEIGHT_PATH} \
    ${DISC_DATA}

WEIGHT_PATH=v44/train_0314_163405/model.pth  # set the path to the trained model
python v46.py \
    -a vit_base_r50_s16_224_in21k --seed 34722999 --workers 8 \
    --epochs 3 --lr 0.0005 --wd 1e-6 --optimizer sgd --batch-size 16 --val-batch-size 16 \
    --pos-margin 0.1 --neg-margin 1.1 --input-size 416 --memory-size 16384  --sample_size 2000000 --feature_dim 512 \
    --ddp_strategy deepspeed_stage_2 --warmup_steps 0 --num_frames 32 --eval_subset \
    --query_video_dir ${COMPETITION_DATA}/train/query \
    --ref_video_dir ${COMPETITION_DATA}/train/reference \
    --noise_video_dir ${COMPETITION_DATA}/test/reference \
    --eval_gt_path ${COMPETITION_DATA}/train/train_matching_ground_truth.csv \
    --query_metadata_path ${COMPETITION_DATA}/train/train_query_metadata.csv \
    --ref_metadata_path ${COMPETITION_DATA}/train/train_reference_metadata.csv \
    --noise_metadata_path ${COMPETITION_DATA}/test/test_reference_metadata.csv \
    --weight ${WEIGHT_PATH} \
    ${DISC_DATA}

WEIGHT_PATH=v46/train_0315_014442/model.pth  # set the path to the trained model
python v45.py \
  -a vit_base_r50_s16_224_in21k  --seed 5699999 --workers 8 \
  --epochs 5 --lr 0.0005 --wd 1e-6 --batch-size 16 --val-batch-size 8 \
  --pos-margin 0.1 --neg-margin 1.1 --input-size 448 --memory-size 0 --feature_dim 512 \
  --ddp_strategy deepspeed_stage_2 --warmup_steps 300 --num_frames 32 --eval_subset --num_negatives 2 \
  --query_video_dir ${COMPETITION_DATA}/train/query \
  --ref_video_dir ${COMPETITION_DATA}/train/reference \
  --noise_video_dir ${COMPETITION_DATA}/test/reference \
  --eval_gt_path ${COMPETITION_DATA}/train/train_matching_ground_truth.csv \
  --query_metadata_path ${COMPETITION_DATA}/train/train_query_metadata.csv \
  --ref_metadata_path ${COMPETITION_DATA}/train/train_reference_metadata.csv \
  --noise_metadata_path ${COMPETITION_DATA}/test/test_reference_metadata.csv \
  --weight ${WEIGHT_PATH} \
  ${DISC_DATA}
```

## Inference

```
WEIGHT_PATH=v45/train_0315_085057/model.pth  # set the path to the trained model
python inference.py \
    -a vit_base_r50_s16_224_in21k --workers 8 \
    --val-batch-size 1 --input-size 448 --feature_dim 512 \
    --fps 1 --num_views 5 --len_cap 42 \
    --ddp_strategy ddp_find_unused_parameters_false \
    --query_video_dir ${COMPETITION_DATA}/train/query \
    --ref_video_dir ${COMPETITION_DATA}/train/reference \
    --test_query_video_dir ${COMPETITION_DATA}/test/query \
    --test_ref_video_dir ${COMPETITION_DATA}/test/reference \
    --gt_path ${COMPETITION_DATA}/train/train_matching_ground_truth.csv \
    --pred_output_dir inference_output --weight ${WEIGHT_PATH}
```
