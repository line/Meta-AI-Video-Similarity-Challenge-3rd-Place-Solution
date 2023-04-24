#!/bin/bash -eu

conda run -n condaenv python prepare.py \
    --video-dir-path /path/to/train_reference/query \
    --augly-metadata-path /path/to/train_reference/cvl_train_augly_metadata

conda run -n condaenv python train.py \
  --dataset-dir-path ./dataset \
  --frames-per-video 5 \
  --model-name hf-hub:timm/convnext_base.clip_laion2b_augreg_ft_in1k \
  --epochs 8 \
  --batch-size 64 \
  --devices 0 1 2 3 \
  --num-workers 8 \
  --accelerator gpu \
  --ddp-strategy ddp_find_unused_parameters_false 
