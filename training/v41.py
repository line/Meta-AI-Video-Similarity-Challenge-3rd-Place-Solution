"""
Copyright 2023 LINE Corporation

LINE Corporation licenses this file to you under the Apache License,
version 2.0 (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.
"""
from __future__ import annotations

import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from augly.image import (
    EncodingQuality,
    OneOf,
    RandomBlur,
    RandomEmojiOverlay,
    RandomNoise,
    RandomPixelization,
    RandomRotation,
)
from augly.image.functional import overlay_image
from augly.image.transforms import BaseTransform
from isc_feature_extractor.model import ISCNet
from loguru import logger
from PIL import Image
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import NativeSyncBatchNorm
from pytorch_metric_learning import losses
from pytorch_metric_learning.utils import distributed as pml_dist
from torchvision.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomErasing,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomPerspective,
    RandomResizedCrop,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)
from utils import (
    CustomModelCheckpoint,
    CustomNormalize,
    CustomWriter,
    NCropsTransform,
    RandomOverlayText,
    ShuffledAug,
    VSCLitDataModule,
    collate_fn_for_variable_size,
    convert2rgb,
    cosine_scheduler,
    decord_clip_reader,
    knn_search,
)
from vsc.baseline.score_normalization import score_normalize
from vsc.index import VideoFeature
from vsc.metrics import (
    CandidatePair,
    Match,
    average_precision,
)
from vsc.storage import store_features

ver = __file__.replace(".py", "")


class ISCNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        fc_dim: int = 256,
        l2_normalize=True,
    ):
        super().__init__()

        self.backbone = backbone
        self.fc = nn.Linear(self.backbone.num_features, fc_dim, bias=False)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        self.l2_normalize = l2_normalize

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.fc(x)
        x = self.bn(x)
        if self.l2_normalize:
            x = F.normalize(x)
        return x


class ISCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        paths: list[Path],
        transforms,
    ):
        self.paths = paths
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.paths[i])
        image = self.transforms(image)
        return i, image


class VSCValDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: Path,
        ids: list[str],
        transform: Callable | None = None,
        count: int = 32,
    ):
        self.data_dir = data_dir
        self.ids = ids
        self.transform = transform
        self.count = count

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        path = self.data_dir / f"{id_}.mp4"
        clip, frame_pos = decord_clip_reader(
            path=path,
            start=0,
            duration=-1,
            count=self.count,
            tensor_format="TCHW",
        )
        if self.transform is not None:
            clip = self.transform(clip)
        video_id = torch.full(
            size=(self.count,),
            fill_value=int(id_[1:]),
            dtype=torch.long,
        )
        return clip, video_id, frame_pos


class VSCLitModule(pl.LightningModule):
    def __init__(
        self,
        args,
        model,
        loss_fn,
        train_transform: Callable | None = None,
        val_transform: Callable | None = None,
        multi_crops: bool = False,
    ):
        super().__init__()
        self.args = args
        self.model = model
        self.loss_fn = loss_fn
        self.train_transform = train_transform
        if self.train_transform is not None:
            self.train_transform = train_transform.to(torch.cuda.current_device())
        self.val_transform = val_transform
        if self.val_transform is not None:
            self.val_transform = val_transform.to(torch.cuda.current_device())
        self.multi_crops = multi_crops

    def training_step(self, batch, batch_idx):
        labels, images = batch
        images = torch.cat([image for image in images], dim=0).half()
        labels = torch.tile(labels, dims=(2,))

        embeddings = self.model(images)
        loss = self.loss_fn(embeddings, labels)
        self.log_dict({"train_loss": loss}, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx):
        clip, video_ids, frame_pos = batch

        if self.val_transform is not None:
            if self.multi_crops:
                all_crops = []
                for _clip in clip:  # _clip: (T, C, H, W)
                    crops = [
                        _clip,
                        _clip[:, :, : _clip.shape[-2] // 2, :],
                        _clip[:, :, _clip.shape[-2] // 2 :],
                        _clip[:, :, :, : _clip.shape[-1] // 2],
                        _clip[:, :, :, _clip.shape[-1] // 2 :],
                    ]
                    crops = torch.cat([self.val_transform(_) for _ in crops], dim=0)
                    all_crops.append(crops)
                clip = torch.stack(all_crops)
                video_ids = torch.stack(video_ids)
            else:
                clip = torch.stack([self.val_transform(_) for _ in clip])
                video_ids = torch.stack(video_ids)

        b, t, c, h, w = clip.shape
        clip = clip.reshape(-1, c, h, w)
        clip = clip.half()
        embeddings = self.model(clip)
        return embeddings, video_ids, frame_pos

    def configure_optimizers(self):
        def is_no_decay_param(n, p):
            return p.ndim < 2 or "norm" in n or "bias" in n or "gain" in n

        # # TODO
        # for name, param in self.model.backbone.named_parameters():
        #     if 'stages_3' in name:
        #         continue
        #     param.requires_grad = False

        decay = []
        no_decay = []
        # added_params = []
        added_param_names = [
            "model.fc.weight",
            "model.bn.weight",
            "model.bn.bias",
        ]
        added_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if name in added_param_names:
                added_params.append(param)
            else:
                if is_no_decay_param(name, param):
                    # rank_zero_info(f'no_decay: {name}')
                    no_decay.append(param)
                else:
                    # rank_zero_info(f'decay: {name}')
                    decay.append(param)

        # https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/mmaction/core/optimizer/tsm_optimizer_constructor.py
        optim_params = [
            {"params": no_decay, "lr": self.args.lr, "weight_decay": 0.0},
            {"params": decay, "lr": self.args.lr, "weight_decay": self.args.wd},
            {
                "params": added_params,
                "lr": self.args.lr * 5,
                "weight_decay": self.args.wd * 10,
            },
        ]
        if self.args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(optim_params)
        elif self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(optim_params, momentum=args.momentum)
        else:
            raise ValueError(f"Unknown optimizer: {self.args.optimizer}")

        if self.args.warmup_steps > 0:
            num_training_steps = (
                len(self.trainer.datamodule.train_dataloader()) * self.args.epochs
            )
            scheduler = {
                "scheduler": cosine_scheduler(
                    optimizer,
                    training_steps=num_training_steps,
                    warmup_steps=self.args.warmup_steps,
                ),
                "name": "learning_rate",
                "interval": "step",
            }
            return [optimizer], [scheduler]
        else:
            return [optimizer]


class RandomCompositeImageAndResizedCrop(BaseTransform):
    def __init__(
        self,
        img_paths: list[Path],
        opacity_lower: float = 0.5,
        size_lower: float = 0.4,
        size_upper: float = 0.6,
        input_size: int = 224,
        moderate_scale_lower: float = 0.7,
        hard_scale_lower: float = 0.15,
        overlay_p: float = 0.05,
        alpha_blending_p: float = 0.05,
        horizontal_stacking_p: float = 0.05,
        vertical_stacking_p: float = 0.05,
        p: float = 1.0,
    ):
        super().__init__(p)
        self.img_paths = img_paths
        self.opacity_lower = opacity_lower
        self.size_lower = size_lower
        self.size_upper = size_upper
        self.input_size = input_size
        self.moderate_scale_lower = moderate_scale_lower
        self.hard_scale_lower = hard_scale_lower
        self.overlay_p = overlay_p
        self.alpha_blending_p = alpha_blending_p
        self.horizontal_stacking_p = horizontal_stacking_p
        self.vertical_stacking_p = vertical_stacking_p

    def apply_transform(
        self,
        image: Image.Image,
        metadata: list[dict[str, Any]] | None = None,
        bboxes: list[tuple] | None = None,
        bbox_format: str | None = None,
    ) -> Image.Image:
        path = random.choice(self.img_paths)

        choices = [
            "overlay",
            "alpha_blending",
            "horizontal_stacking",
            "vertical_stacking",
            "nothing",
        ]
        weights = [
            self.overlay_p,
            self.alpha_blending_p,
            self.horizontal_stacking_p,
            self.vertical_stacking_p,
        ]
        weights.append(1.0 - sum(weights))

        composite_way = random.choices(choices, weights, k=1)[0]
        if composite_way == "overlay":
            if random.uniform(0.0, 1.0) > 0.5:
                background = Image.open(path)
                overlay = image
            else:
                background = image
                overlay = Image.open(path)
            overlay_size = random.uniform(self.size_lower, self.size_upper)
            image = overlay_image(
                background,
                overlay=overlay,
                opacity=random.uniform(self.opacity_lower, 1.0),
                overlay_size=overlay_size,
                x_pos=random.uniform(0.0, 1.0 - overlay_size),
                y_pos=random.uniform(0.0, 1.0 - overlay_size),
                metadata=metadata,
            )
            return RandomResizedCrop(
                self.input_size, scale=(self.moderate_scale_lower, 1.0)
            )(image)

        elif composite_way == "alpha_blending":
            blended_image = Image.open(path)
            image = overlay_image(
                image,
                overlay=blended_image,
                opacity=0.5,
                overlay_size=1.0,
                x_pos=0.0,
                y_pos=0.0,
                metadata=metadata,
            )
            return RandomResizedCrop(
                self.input_size, scale=(self.moderate_scale_lower, 1.0)
            )(image)

        elif composite_way == "vertical_stacking":
            stacked_image = Image.open(path)
            stacked_image = stacked_image.resize((image.width, image.height))
            # factor = image.width / stacked_image.width
            # stacked_image = stacked_image.resize((image.width, int(stacked_image.height * factor)))
            image = Image.fromarray(
                np.concatenate([np.array(image), np.array(stacked_image)], axis=0)
            )
            return RandomResizedCrop(
                self.input_size, scale=(self.moderate_scale_lower, 1.0)
            )(image)

        elif composite_way == "horizontal_stacking":
            stacked_image = Image.open(path)
            stacked_image = stacked_image.resize((image.width, image.height))
            # factor = image.height / stacked_image.height
            # stacked_image = stacked_image.resize((int(stacked_image.width * factor), image.height))
            image = Image.fromarray(
                np.concatenate([np.array(image), np.array(stacked_image)], axis=1)
            )
            return RandomResizedCrop(
                self.input_size, scale=(self.moderate_scale_lower, 1.0)
            )(image)

        elif composite_way == "nothing":
            return RandomResizedCrop(
                self.input_size, scale=(self.hard_scale_lower, 1.0)
            )(image)

        else:
            raise ValueError("Invalid composite way")


def train(args):
    cudnn.benchmark = True
    pl.seed_everything(args.seed, workers=True)

    # if args.dryrun:
    #     sample_size = 32
    #     args.epochs = 1
    # else:
    #     sample_size = 9999999

    input_size = tuple(map(int, args.input_size.split("x")))
    if len(input_size) == 1:
        input_size = (input_size[0], input_size[0])

    backbone = timm.create_model(
        args.arch, num_classes=0, pretrained=True, img_size=input_size
    )
    model = ISCNet(
        backbone=backbone,
        fc_dim=args.feature_dim,
        l2_normalize=args.l2_normalize,
    )
    data_config = timm.data.resolve_data_config(args={}, model=backbone)

    if args.weight is not None:
        from timm.layers import resample_abs_pos_embed

        weight = torch.load(args.weight, map_location="cpu")
        weight["backbone.pos_embed"] = resample_abs_pos_embed(
            weight["backbone.pos_embed"].float(),
            new_size=[input_size[0] // 16, input_size[1] // 16],
        ).half()
        model.load_state_dict(weight)

    loss_fn = losses.ContrastiveLoss(
        pos_margin=args.pos_margin, neg_margin=args.neg_margin
    )
    if args.memory_size > 0:
        loss_fn = losses.CrossBatchMemory(
            loss_fn, embedding_size=args.feature_dim, memory_size=args.memory_size
        )
    loss_fn = pml_dist.DistributedLossWrapper(loss=loss_fn, efficient=args.efficient)

    train_paths = list((Path(args.data) / "train_images").glob("**/*.jpg")) + list(
        (Path(args.data) / "reference_images").glob("**/*.jpg")
    )
    train_paths = train_paths[: args.sample_size]
    print("training sample size:", len(train_paths))

    normalize_rgb = Normalize(mean=data_config["mean"], std=data_config["std"])
    aug_moderate = [
        RandomResizedCrop(input_size, scale=(0.7, 1.0)),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize_rgb,
    ]
    aug_list = [
        ColorJitter(0.3, 0.3, 0.3, 0.1),
        RandomPixelization(p=0.1),
        OneOf([EncodingQuality(quality=q) for q in [10, 20, 30, 50]], p=0.1),
        RandomGrayscale(p=0.05),
        RandomBlur(p=0.1),
        RandomPerspective(p=0.1),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.05),
        RandomOverlayText(p=0.1),
        RandomEmojiOverlay(
            p=0.05, emoji_size=(0.1, 0.3), x_pos=(0.0, 1.0), y_pos=(0.0, 1.0), seed=None
        ),
        RandomNoise(p=0.1),
    ]
    aug_hard = [
        RandomRotation(p=0.05),
        RandomCompositeImageAndResizedCrop(
            train_paths,
            opacity_lower=0.6,
            size_lower=0.3,
            size_upper=0.7,
            input_size=input_size,
            moderate_scale_lower=0.8,
            hard_scale_lower=0.5,
            overlay_p=0.05,
            alpha_blending_p=0.05,
            horizontal_stacking_p=0.0,
            vertical_stacking_p=0.0,
            p=1.0,
        ),
        ShuffledAug(aug_list),
        convert2rgb,
        ToTensor(),
        RandomErasing(value="random", p=0.05),
        normalize_rgb,
    ]

    normalize_rgb = CustomNormalize(mean=data_config["mean"], std=data_config["std"])
    t_val = nn.Sequential(
        Resize(size=input_size),
        normalize_rgb,
    )
    t_val = torch.jit.script(t_val)

    train_dataset = ISCDataset(
        paths=train_paths,
        transforms=NCropsTransform(
            Compose(aug_moderate),
            Compose(aug_hard),
            ncrops=2,
        ),
    )

    pl_dm = VSCLitDataModule(
        train_datasets=[train_dataset],
        val_datasets=[],
        train_batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        workers=args.workers,
        # collate_fn=collate_fn_for_variable_size,
    )

    pl_model = VSCLitModule(
        args=args,
        model=model,
        loss_fn=loss_fn,
        val_transform=t_val,
    )

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    train_dir = Path(f"{ver}/train_{timestamp}")
    wandb_logger = WandbLogger(project="vsc", save_dir=train_dir)
    wandb_logger.log_hyperparams(args)
    checkpoint_callback = CustomModelCheckpoint(
        dirpath=wandb_logger.save_dir,
        save_last=True,
        filename="artifacts-{epoch:02d}",
        # monitor="val/segment_ap",
        # save_top_k=1,
        # mode="max",
    )

    trainer = pl.Trainer(
        strategy=args.ddp_strategy,
        accelerator="gpu",
        devices=-1,
        max_epochs=args.epochs,
        check_val_every_n_epoch=9999999,
        precision=16,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        plugins=[NativeSyncBatchNorm()],
    )
    trainer.fit(pl_model, pl_dm)

    args.pred_output_dir = trainer.log_dir
    ap = predict(args, pl_model=pl_model)
    if trainer.global_rank == 0:
        wandb_logger.log_metrics({"uAP": ap.ap})
        checkpoint_callback._save_checkpoint(trainer, wandb_logger.save_dir)


def predict(args, pl_model=None):
    pred_output_dir = Path(args.pred_output_dir)

    if args.dryrun:
        sample_size = 100
    else:
        sample_size = 9999999

    gt_matches = Match.read_csv(args.eval_gt_path, is_gt=True)
    gt_matches = gt_matches[:sample_size]

    if args.eval_subset:
        gt = pd.read_csv(args.eval_gt_path)
        query_subset_ids = sorted(gt["query_id"].unique().tolist())
        ref_subset_ids = sorted(gt["ref_id"].unique().tolist())
        noise_size = 1024
        noise_subset_ids = [
            d.stem
            for d in sorted(Path(args.noise_video_dir).glob("*.mp4"))[:noise_size]
        ]
        query_ids = query_subset_ids
        ref_ids = ref_subset_ids
        noise_ids = noise_subset_ids
    else:
        query_ids = pd.read_csv(args.query_metadata_path, usecols=["video_id"])[
            "video_id"
        ].tolist()
        ref_ids = pd.read_csv(args.ref_metadata_path, usecols=["video_id"])[
            "video_id"
        ].tolist()
        noise_ids = pd.read_csv(args.noise_metadata_path, usecols=["video_id"])[
            "video_id"
        ].tolist()

    input_size = tuple(map(int, args.input_size.split("x")))
    if len(input_size) == 1:
        input_size = (input_size[0], input_size[0])

    backbone = timm.create_model(
        args.arch, num_classes=0, pretrained=True, img_size=input_size
    )
    model = ISCNet(
        backbone=backbone,
        fc_dim=args.feature_dim,
        l2_normalize=args.l2_normalize,
    )
    data_config = timm.data.resolve_model_data_config(backbone)

    normalize_rgb = CustomNormalize(mean=data_config["mean"], std=data_config["std"])
    t_val = nn.Sequential(
        Resize(size=input_size),
        normalize_rgb,
    )
    t_val = torch.jit.script(t_val)

    pl_dm = VSCLitDataModule(
        train_datasets=[
            # dummy
            VSCValDataset(
                data_dir=Path(args.query_video_dir),
                ids=query_ids[:sample_size],
                count=args.num_frames,
            )
        ],
        val_datasets=[
            VSCValDataset(
                data_dir=Path(args.query_video_dir),
                ids=query_ids[:sample_size],
                count=args.num_frames,
            ),
            VSCValDataset(
                data_dir=Path(args.ref_video_dir),
                ids=ref_ids[:sample_size],
                count=args.num_frames,
            ),
            VSCValDataset(
                data_dir=Path(args.noise_video_dir),
                ids=noise_ids[:sample_size],
                count=args.num_frames,
            ),
        ],
        train_batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        workers=args.workers,
        collate_fn=collate_fn_for_variable_size,
    )

    if pl_model is None:
        if args.weight is not None:
            state_dict = torch.load(args.weight, map_location="cpu")
            model.load_state_dict(state_dict)
        pl_model = VSCLitModule(
            args=args,
            model=model,
            loss_fn=None,
            val_transform=t_val,
            multi_crops=args.multi_crops,
        )

    pred_writer = CustomWriter(output_dir=pred_output_dir, write_interval="epoch")
    trainer = pl.Trainer(
        gpus=-1,
        strategy=args.ddp_strategy,
        accelerator="gpu",
        precision=16,
        num_sanity_val_steps=0,
        callbacks=[pred_writer],
    )
    trainer.predict(
        pl_model, dataloaders=pl_dm.val_dataloader(), return_predictions=False
    )

    torch.distributed.barrier()

    if trainer.global_rank == 0:
        return aggregate_preds_and_evaluate(
            args, pred_output_dir, gt_matches, query_ids, ref_ids, noise_ids
        )


def aggregate_preds_and_evaluate(
    args, pred_output_dir, gt_matches, query_ids, ref_ids, noise_ids
):
    batch_indices_paths = sorted(pred_output_dir.glob("batch_indices*"))
    pred_paths = sorted(pred_output_dir.glob("predictions*"))

    num_views = 5 if args.multi_crops else 1
    num_frames = args.num_frames * num_views

    query_all_batch_indices = []
    query_all_preds = []
    query_all_frame_pos = []
    ref_all_batch_indices = []
    ref_all_preds = []
    ref_all_frame_pos = []
    noise_all_batch_indices = []
    noise_all_preds = []
    noise_all_frame_pos = []
    for batch_indices_path, pred_path in zip(batch_indices_paths, pred_paths):
        batch_indices = torch.load(batch_indices_path)
        predictions = torch.load(pred_path)

        flat_batch_indices = sum(batch_indices[0], [])
        embeddings, video_id, frame_pos = list(zip(*predictions[0]))
        flat_predictions = torch.cat(embeddings)
        flat_predictions = flat_predictions.reshape(
            -1, num_frames, flat_predictions.shape[-1]
        )
        flat_frame_pos = np.concatenate([np.stack(f) for f in frame_pos], axis=0)
        query_all_batch_indices.extend(flat_batch_indices)
        query_all_preds.append(flat_predictions)
        query_all_frame_pos.append(flat_frame_pos)

        flat_batch_indices = sum(batch_indices[1], [])
        embeddings, video_id, frame_pos = list(zip(*predictions[1]))
        flat_predictions = torch.cat(embeddings)
        flat_predictions = flat_predictions.reshape(
            -1, num_frames, flat_predictions.shape[-1]
        )
        flat_frame_pos = np.concatenate([np.stack(f) for f in frame_pos], axis=0)
        ref_all_batch_indices.extend(flat_batch_indices)
        ref_all_preds.append(flat_predictions)
        ref_all_frame_pos.append(flat_frame_pos)

        flat_batch_indices = sum(batch_indices[2], [])
        embeddings, video_id, frame_pos = list(zip(*predictions[2]))
        flat_predictions = torch.cat(embeddings)
        flat_predictions = flat_predictions.reshape(
            -1, num_frames, flat_predictions.shape[-1]
        )
        flat_frame_pos = np.concatenate([np.stack(f) for f in frame_pos], axis=0)
        noise_all_batch_indices.extend(flat_batch_indices)
        noise_all_preds.append(flat_predictions)
        noise_all_frame_pos.append(flat_frame_pos)

        os.remove(batch_indices_path)
        os.remove(pred_path)

    query_all_batch_indices = torch.tensor(query_all_batch_indices)
    query_all_preds = torch.cat(query_all_preds)
    sorted_query_all_preds = query_all_preds[query_all_batch_indices.argsort()]
    query_all_frame_pos = np.concatenate(query_all_frame_pos, axis=0)
    sorted_query_all_frame_pos = query_all_frame_pos[query_all_batch_indices.argsort()]

    ref_all_batch_indices = torch.tensor(ref_all_batch_indices)
    ref_all_preds = torch.cat(ref_all_preds)
    sorted_ref_all_preds = ref_all_preds[ref_all_batch_indices.argsort()]
    ref_all_frame_pos = np.concatenate(ref_all_frame_pos, axis=0)
    sorted_ref_all_frame_pos = ref_all_frame_pos[ref_all_batch_indices.argsort()]

    noise_all_batch_indices = torch.tensor(noise_all_batch_indices)
    noise_all_preds = torch.cat(noise_all_preds)
    sorted_noise_all_preds = noise_all_preds[noise_all_batch_indices.argsort()]
    noise_all_frame_pos = np.concatenate(noise_all_frame_pos, axis=0)
    sorted_noise_all_frame_pos = noise_all_frame_pos[noise_all_batch_indices.argsort()]

    def _construct_video_features(video_ids, embeddings, timestamps):
        video_features = []
        for video_id, embedding, timestamp in zip(video_ids, embeddings, timestamps):
            video_feature = VideoFeature(
                video_id=video_id,
                timestamps=timestamp,
                feature=embedding.cpu().float().numpy(),
            )
            video_features.append(video_feature)
        return video_features

    if args.multi_crops:
        sorted_query_all_frame_pos = np.concatenate(
            [sorted_query_all_frame_pos for _ in range(num_views)], axis=1
        )
        sorted_ref_all_frame_pos = np.concatenate(
            [sorted_ref_all_frame_pos for _ in range(num_views)], axis=1
        )
        sorted_noise_all_frame_pos = np.concatenate(
            [sorted_noise_all_frame_pos for _ in range(num_views)], axis=1
        )

    video_features = {
        "query": _construct_video_features(
            video_ids=query_ids,
            embeddings=sorted_query_all_preds,
            timestamps=sorted_query_all_frame_pos,
        ),
        "ref": _construct_video_features(
            video_ids=ref_ids,
            embeddings=sorted_ref_all_preds,
            timestamps=sorted_ref_all_frame_pos,
        ),
        "noise": _construct_video_features(
            video_ids=noise_ids,
            embeddings=sorted_noise_all_preds,
            timestamps=sorted_noise_all_frame_pos,
        ),
    }
    for split, vfs in video_features.items():
        store_features(pred_output_dir / f"{split}.npz", vfs)

    queries, refs = score_normalize(
        video_features["query"],
        video_features["ref"],
        video_features["noise"],
        beta=1.2,
    )

    candidates = knn_search(queries, refs, retrieve_per_query=25, use_gpu=True)
    CandidatePair.write_csv(candidates, pred_output_dir / "candidates.csv")
    print("candidates saved:", pred_output_dir / "candidates.csv")

    ap = average_precision(CandidatePair.from_matches(gt_matches), candidates)
    logger.info(f"AP: {ap.ap:.4f}")
    return ap


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VSC Image Level Training")
    parser.add_argument("data", metavar="DIR", help="path to dataset")
    parser.add_argument(
        "--query_video_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ref_video_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--noise_video_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--eval_gt_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--query_metadata_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ref_metadata_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--noise_metadata_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-a", "--arch", metavar="ARCH", default="swin_base_patch4_window7_224"
    )
    parser.add_argument("--workers", default=os.cpu_count(), type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("-b", "--batch-size", default=512, type=int)
    parser.add_argument("--val-batch-size", default=1024, type=int)
    parser.add_argument("--lr", default=0.05, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--wd", default=1e-4, type=float)
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument("--gem-p", default=1.0, type=float)
    parser.add_argument("--gem-eval-p", default=1.0, type=float)
    parser.add_argument("--mode", default="train", type=str, help="train or extract")
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--pos-margin", default=0.0, type=float)
    parser.add_argument("--neg-margin", default=0.7, type=float)
    parser.add_argument("--input-size", default=224)
    parser.add_argument("--weight", type=str)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--memory-size", default=1024, type=int)
    parser.add_argument("--feature_dim", default=512, type=int)
    parser.add_argument("--l2_normalize", type=bool, default=True)
    parser.add_argument("--ddp_strategy", default="deepspeed_stage_2", type=str)
    parser.add_argument("--num_frames", default=32, type=int)
    parser.add_argument("--frozen_stages", default=-1, type=int)
    parser.add_argument("--warmup_steps", default=100, type=int)
    parser.add_argument("--pred_output_dir", type=str, default=f"{ver}/preds")
    parser.add_argument("--efficient", action="store_true")
    parser.add_argument("--eval_subset", action="store_true")
    parser.add_argument("--multi_crops", action="store_true")
    parser.add_argument("--sample_size", default=9999999, type=int)
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    if args.mode == "predict":
        predict(args)
