# モデル学習の実行コマンド

## blending特化モデル
CUDA_VISIBLE_DEVICES=4,5,6,7 python v47.py \
    -a vit_base_r50_s16_224.orig_in21k --seed 34722999 --workers 8 \
    --epochs 3 --lr 0.0005 --wd 1e-6 --optimizer sgd --batch-size 32 --val-batch-size 16 \
    --pos-margin 0.1 --neg-margin 1.1 --input-size 448 --memory-size 1024  --sample_size 100000 --feature_dim 512 \
    --ddp_strategy deepspeed_stage_2 --warmup_steps 0 \
    --query_video_dir /app/data/vsc/competition_data/train/query \
    --ref_video_dir /app/data/vsc/competition_data/train/reference \
    --noise_video_dir /app/data/vsc/competition_data/test/reference \
    --eval_gt_path /share-cvlab/yokoo/vsc/competition_data/train/train_matching_ground_truth.csv \
    --query_metadata_path /share-cvlab/yokoo/vsc/competition_data/train/train_query_metadata.csv \
    --ref_metadata_path /share-cvlab/yokoo/vsc/competition_data/train/train_reference_metadata.csv \
    --noise_metadata_path /share-cvlab/yokoo/vsc/competition_data/test/test_reference_metadata.csv \
    --num_frames 32 --eval_subset \
    --weight v44/train_0314_163405/model.pth \
    /app/data/isc/
AP: 0.8746

python v47.py \
    -a vit_base_r50_s16_224.orig_in21k --seed 34722999 --workers 8 \
    --epochs 3 --lr 0.0005 --wd 1e-6 --optimizer sgd --batch-size 32 --val-batch-size 16 \
    --pos-margin 0.1 --neg-margin 1.1 --input-size 448 --memory-size 1024  --sample_size 2000000 --feature_dim 512 \
    --ddp_strategy deepspeed_stage_2 --warmup_steps 0 \
    --query_video_dir /app/data/vsc/competition_data/train/query \
    --ref_video_dir /app/data/vsc/competition_data/train/reference \
    --noise_video_dir /app/data/vsc/competition_data/test/reference \
    --eval_gt_path /share-cvlab/yokoo/vsc/competition_data/train/train_matching_ground_truth.csv \
    --query_metadata_path /share-cvlab/yokoo/vsc/competition_data/train/train_query_metadata.csv \
    --ref_metadata_path /share-cvlab/yokoo/vsc/competition_data/train/train_reference_metadata.csv \
    --noise_metadata_path /share-cvlab/yokoo/vsc/competition_data/test/test_reference_metadata.csv \
    --num_frames 32 --eval_subset \
    --weight v44/train_0314_163405/model.pth \
    /app/data/isc/
v47/train_0322_094308/model.pth
sudo cp /yokoo-data/code/vsc/yokoo/v47/train_0322_094308/model.pth /yokoo-data/code/meta-vsc-descriptor-runtime/submission_src/model_assets/train_0322_094308_model.pth
AP: 0.8852

## Hybrid | DISC(SSL)

CUDA_VISIBLE_DEVICES=4,5,6,7 python isc_test_length_aware_tta.py \
    -a vit_base_r50_s16_224_in21k --workers 8 \
    --val-batch-size 1 --input-size 448 --feature_dim 512 \
    --fps 1 --num_views 5 --len_cap 42 \
    --ddp_strategy ddp_find_unused_parameters_false \
    --query_video_dir /app/data/vsc/competition_data/train/query \
    --ref_video_dir /app/data/vsc/competition_data/train/reference \
    --test_query_video_dir /app/data/vsc/competition_data/test/query \
    --test_ref_video_dir /app/data/vsc/competition_data/test/reference \
    --gt_path /share-cvlab/yokoo/vsc/competition_data/train/train_matching_ground_truth.csv \
    --pred_output_dir hybrid_preds/1fps_5view_42cap --weight v45/train_0315_085057/model.pth

python v45.py \
  -a vit_base_r50_s16_224.orig_in21k  --seed 5699999 --workers 8 \
  --epochs 5 --lr 0.0005 --wd 1e-6 --batch-size 16 --val-batch-size 8 \
  --pos-margin 0.1 --neg-margin 1.1 \
  --input-size 448 --memory-size 0 --feature_dim 512 \
  --ddp_strategy deepspeed_stage_2 --warmup_steps 300 \
  --query_video_dir /app/data/vsc/competition_data/train/query \
  --ref_video_dir /app/data/vsc/competition_data/train/reference \
  --noise_video_dir /app/data/vsc/competition_data/test/reference \
  --eval_gt_path /share-cvlab/yokoo/vsc/competition_data/train/train_matching_ground_truth.csv \
  --query_metadata_path /share-cvlab/yokoo/vsc/competition_data/train/train_query_metadata.csv \
  --ref_metadata_path /share-cvlab/yokoo/vsc/competition_data/train/train_reference_metadata.csv \
  --noise_metadata_path /share-cvlab/yokoo/vsc/competition_data/test/test_reference_metadata.csv \
  --num_frames 32 --eval_subset --num_negatives 2 \
  --weight v46/train_0315_014442/model.pth \
  /app/data/isc/
candidates saved: /app/yokoo/v45/train_0315_085057/candidates.csv
2023-03-15 09:10:18.687 | INFO     | __main__:aggregate_preds_and_evaluate:785 - AP: 0.8886

python v46.py \
    -a vit_base_r50_s16_224.orig_in21k --seed 34722999 --workers 8 \
    --epochs 3 --lr 0.0005 --wd 1e-6 --optimizer sgd --batch-size 16 --val-batch-size 16 \
    --pos-margin 0.1 --neg-margin 1.1 --input-size 416 --memory-size 16384  --sample_size 2000000 --feature_dim 512 \
    --ddp_strategy deepspeed_stage_2 --warmup_steps 0 \
    --query_video_dir /app/data/vsc/competition_data/train/query \
    --ref_video_dir /app/data/vsc/competition_data/train/reference \
    --noise_video_dir /app/data/vsc/competition_data/test/reference \
    --eval_gt_path /share-cvlab/yokoo/vsc/competition_data/train/train_matching_ground_truth.csv \
    --query_metadata_path /share-cvlab/yokoo/vsc/competition_data/train/train_query_metadata.csv \
    --ref_metadata_path /share-cvlab/yokoo/vsc/competition_data/train/train_reference_metadata.csv \
    --noise_metadata_path /share-cvlab/yokoo/vsc/competition_data/test/test_reference_metadata.csv \
    --num_frames 32 --eval_subset \
    --weight v44/train_0314_163405/model.pth \
    /app/data/isc/
candidates saved: /app/yokoo/v46/train_0315_014442/candidates.csv
2023-03-15 08:13:51.299 | INFO     | __main__:aggregate_preds_and_evaluate:816 - AP: 0.8714

python v44.py \
    -a vit_base_r50_s16_224.orig_in21k --seed 722999 --workers 8 \
    --epochs 5 --lr 0.0005 --wd 1e-6 --optimizer sgd --batch-size 16 --val-batch-size 16 \
    --pos-margin 0.1 --neg-margin 1.1 --input-size 384 --memory-size 8192  --sample_size 2000000 --feature_dim 512 \
    --ddp_strategy deepspeed_stage_2 --warmup_steps 0 \
    --query_video_dir /app/data/vsc/competition_data/train/query \
    --ref_video_dir /app/data/vsc/competition_data/train/reference \
    --noise_video_dir /app/data/vsc/competition_data/test/reference \
    --eval_gt_path /share-cvlab/yokoo/vsc/competition_data/train/train_matching_ground_truth.csv \
    --query_metadata_path /share-cvlab/yokoo/vsc/competition_data/train/train_query_metadata.csv \
    --ref_metadata_path /share-cvlab/yokoo/vsc/competition_data/train/train_reference_metadata.csv \
    --noise_metadata_path /share-cvlab/yokoo/vsc/competition_data/test/test_reference_metadata.csv \
    --num_frames 32 --eval_subset \
    --weight v41/train_0314_024117/model.pth \
    /app/data/isc/
candidates saved: /app/yokoo/v44/train_0314_163405/candidates.csv
2023-03-15 01:24:14.778 | INFO     | __main__:aggregate_preds_and_evaluate:817 - AP: 0.8592

CUDA_VISIBLE_DEVICES=0,1,2,3 /app/yokoo/v41.py -a vit_base_r50_s16_224.orig_in21k --seed 22999 --workers 8 --epochs 10 --lr 0.001 --wd 1e-6 --optimizer sgd --batch-size 32 --val-batch-size 16 --pos-margin 0.0 --neg-margin 1.0 --input-size 224 --memory-size 4096 --feature_dim 512 --sample_size 2000000 --ddp_strategy deepspeed_stage_2 --warmup_steps 0 --query_video_dir /app/data/vsc/competition_data/train/query --ref_video_dir /app/data/vsc/competition_data/train/reference --noise_video_dir /app/data/vsc/competition_data/test/reference --eval_gt_path /share-cvlab/yokoo/vsc/competition_data/train/train_matching_ground_truth.csv --query_metadata_path /share-cvlab/yokoo/vsc/competition_data/train/train_query_metadata.csv --ref_metadata_path /share-cvlab/yokoo/vsc/competition_data/train/train_reference_metadata.csv --noise_metadata_path /share-cvlab/yokoo/vsc/competition_data/test/test_reference_metadata.csv --num_frames 32 --eval_subset /app/data/isc/
AP: 0.8023

## ISC | DISC(SSL) -> VSC

AP=0.8899, v28/train_0306_165658/model.pth
```
CUDA_VISIBLE_DEVICES=4,5,6,7 python v28.py \
  -a isc_selfsup_v98 --seed 999 --workers 8 \
  --epochs 1 --lr 0.01 --wd 1e-6 --batch-size 16 --val-batch-size 16 \
  --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 384 --memory-size 20000 --feature_dim 512 \
  --ddp_strategy deepspeed_stage_2 --warmup_steps 0 \
  --query_video_dir /app/data/vsc/competition_data/train/query \
  --ref_video_dir /app/data/vsc/competition_data/train/reference \
  --noise_video_dir /app/data/vsc/competition_data/test/reference \
  --eval_gt_path /share-cvlab/yokoo/vsc/competition_data/train/train_matching_ground_truth.csv \
  --query_metadata_path /share-cvlab/yokoo/vsc/competition_data/train/train_query_metadata.csv \
  --ref_metadata_path /share-cvlab/yokoo/vsc/competition_data/train/train_reference_metadata.csv \
  --noise_metadata_path /share-cvlab/yokoo/vsc/competition_data/test/test_reference_metadata.csv \
  --num_frames 32 --eval_subset \
  /app/data/isc/
```

AP=0.9166, v31/train_0307_195710/model.pth
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python v31.py \
  -a tf_efficientnetv2_m_in21ft1k --seed 999999 --workers 8 \
  --epochs 3 --lr 0.001 --wd 1e-6 --batch-size 8 --val-batch-size 4 \
  --input-size 384 --feature_dim 512 \
  --ddp_strategy deepspeed_stage_2 --warmup_steps 0 \
  --query_video_dir /app/data/vsc/competition_data/train/query \
  --ref_video_dir /app/data/vsc/competition_data/train/reference \
  --noise_video_dir /app/data/vsc/competition_data/test/reference \
  --eval_gt_path /share-cvlab/yokoo/vsc/competition_data/train/train_matching_ground_truth.csv \
  --metadata_path /share-cvlab/yokoo/vsc/competition_data/test/test_reference_metadata.csv \
  --train_num_frames 8 --num_frames 32 --eval_subset \
  --loss_type arcface --loss_margin 0.4 --loss_scale 48 --sample_size -1 \
  --weight v28/train_0306_165658/model.pth \
  --vsc_data /app/data/vsc/competition_data/test/reference \
  /app/data/isc/
```

推論
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python isc_test_length_aware.py --mode predict \
    -a tf_efficientnetv2_m_in21ft1k \
    --weight v28/train_0306_165658/model.pth \
    --val-batch-size 1 --fps 1 --input-size 384 --feature_dim 512 \
    --ddp_strategy ddp_find_unused_parameters_false \
    --query_video_dir /app/data/vsc/competition_data/train/query \
    --ref_video_dir /app/data/vsc/competition_data/train/reference \
    --noise_video_dir /app/data/vsc/competition_data/test/reference \
    --multi_crops --pred_output_dir isc_test_length_aware/v31_best_1fps_thresh015_lencap42 \
    /app/data/vsc/competition_data/train/

CUDA_VISIBLE_DEVICES=4,5,6,7 python isc_test_length_aware.py --mode predict \
    -a tf_efficientnetv2_m_in21ft1k \
    --weight v31/train_0307_195710/model.pth \
    --val-batch-size 1 --fps 1 --input-size 384 --feature_dim 512 \
    --ddp_strategy ddp_find_unused_parameters_false \
    --query_video_dir /app/data/vsc/competition_data/train/query \
    --ref_video_dir /app/data/vsc/competition_data/train/reference \
    --noise_video_dir /app/data/vsc/competition_data/test/reference \
    --multi_crops --pred_output_dir isc_test_length_aware/v31_best_1fps_thresh015_lencap42 \
    /app/data/vsc/competition_data/train/

# single crop
CUDA_VISIBLE_DEVICES=4,5,6,7 python isc_test_length_aware.py --mode predict \
    -a tf_efficientnetv2_m_in21ft1k \
    --weight v31/train_0307_195710/model.pth \
    --val-batch-size 1 --fps 1 --input-size 384 --feature_dim 512 \
    --ddp_strategy ddp_find_unused_parameters_false \
    --query_video_dir /app/data/vsc/competition_data/train/query \
    --ref_video_dir /app/data/vsc/competition_data/train/reference \
    --noise_video_dir /app/data/vsc/competition_data/test/reference \
    --pred_output_dir isc_test_length_aware/v31_best_1fps \
    /app/data/vsc/competition_data/train/
```

## ISC | DISC(SSL) -> DISC(gt) -> VSC

```
# AP=0.9009, v15/train_0308_035203/model.pth
python v15.py \
  -a tf_efficientnetv2_m_in21ft1k --seed 99999 --workers 8 \
  --epochs 2 --lr 0.01 --wd 1e-6 --batch-size 16 --val-batch-size 8 \
  --pos-margin 0.1 --neg-margin 1.1 \
  --input-size 384 --memory-size 1000 --feature_dim 512 \
  --ddp_strategy deepspeed_stage_2 --warmup_steps 0 \
  --query_video_dir /app/data/vsc/competition_data/train/query \
  --ref_video_dir /app/data/vsc/competition_data/train/reference \
  --noise_video_dir /app/data/vsc/competition_data/test/reference \
  --eval_gt_path /share-cvlab/yokoo/vsc/competition_data/train/train_matching_ground_truth.csv \
  --query_metadata_path /share-cvlab/yokoo/vsc/competition_data/train/train_query_metadata.csv \
  --ref_metadata_path /share-cvlab/yokoo/vsc/competition_data/train/train_reference_metadata.csv \
  --noise_metadata_path /share-cvlab/yokoo/vsc/competition_data/test/test_reference_metadata.csv \
  --num_frames 32 --eval_subset --num_negatives 2 \
  --weight v28/train_0306_165658/model.pth \
  /app/data/isc/
```

AP: 0.9159, v31/train_0308_074535/model.pth
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python v31.py \
  -a tf_efficientnetv2_m_in21ft1k --seed 999999 --workers 8 \
  --epochs 3 --lr 0.001 --wd 1e-6 --batch-size 8 --val-batch-size 4 \
  --input-size 384 --feature_dim 512 \
  --ddp_strategy deepspeed_stage_2 --warmup_steps 0 \
  --query_video_dir /app/data/vsc/competition_data/train/query \
  --ref_video_dir /app/data/vsc/competition_data/train/reference \
  --noise_video_dir /app/data/vsc/competition_data/test/reference \
  --eval_gt_path /share-cvlab/yokoo/vsc/competition_data/train/train_matching_ground_truth.csv \
  --metadata_path /share-cvlab/yokoo/vsc/competition_data/test/test_reference_metadata.csv \
  --train_num_frames 8 --num_frames 32 --eval_subset \
  --loss_type arcface --loss_margin 0.4 --loss_scale 48 --sample_size -1 \
  --weight v15/train_0308_035203/model.pth \
  --vsc_data /app/data/vsc/competition_data/test/reference \
  /app/data/isc/
```

推論
```
CUDA_VISIBLE_DEVICES=4,5,6,7 python isc_test_length_aware.py --mode predict \
    -a tf_efficientnetv2_m_in21ft1k \
    --weight v15/train_0308_035203/model.pth \
    --val-batch-size 1 --fps 1 --input-size 384 --feature_dim 512 \
    --ddp_strategy ddp_find_unused_parameters_false \
    --query_video_dir /app/data/vsc/competition_data/train/query \
    --ref_video_dir /app/data/vsc/competition_data/train/reference \
    --noise_video_dir /app/data/vsc/competition_data/test/reference \
    --multi_crops --pred_output_dir isc_test_length_aware/v15_1fps \
    /app/data/vsc/competition_data/train/

CUDA_VISIBLE_DEVICES=4,5,6,7 python isc_test_length_aware.py --mode predict \
    -a tf_efficientnetv2_m_in21ft1k \
    --weight v31/train_0308_074535/model.pth \
    --val-batch-size 1 --fps 1 --input-size 384 --feature_dim 512 \
    --ddp_strategy ddp_find_unused_parameters_false \
    --query_video_dir /app/data/vsc/competition_data/train/query \
    --ref_video_dir /app/data/vsc/competition_data/train/reference \
    --noise_video_dir /app/data/vsc/competition_data/test/reference \
    --multi_crops --pred_output_dir isc_test_length_aware/v31_from_v15_1fps \
    /app/data/vsc/competition_data/train/
```
