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
from datetime import datetime
from pathlib import Path
from typing import Callable, NamedTuple

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
import torchvision
import wandb
from loguru import logger
from isc_feature_extractor import create_model
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import NativeSyncBatchNorm
from pytorch_lightning.utilities import rank_zero_only
from pytorch_metric_learning import losses
from pytorch_metric_learning.utils import distributed as pml_dist
from torchvision.models.video import R3D_18_Weights, r3d_18
from torchvision.transforms import Normalize, Resize
from utils import (
    CustomModelCheckpoint,
    CustomWriter,
    VSCLitDataModule,
    collate_fn_for_variable_size,
    cosine_scheduler,
    decord_clip_reader,
    knn_search,
)

from vsc.baseline.score_normalization import score_normalize
from vsc.baseline.sscd_baseline import search
from vsc.index import VideoFeature
from vsc.metrics import CandidatePair, Match, average_precision, format_video_id
from vsc.storage import store_features, same_value_ranges

ver = __file__.replace(".py", "")


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class ISCNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        fc_dim: int = 256,
        p: float = 1.0,
        eval_p: float = 1.0,
        l2_normalize: bool = True,
        device: str = "cuda",
        is_training: bool = False,
    ):
        super().__init__()

        self.backbone = backbone
        if hasattr(backbone, "num_features"):
            self.is_cnn = False
            in_channels = backbone.num_features
        else:
            self.is_cnn = True
            in_channels = backbone.feature_info.info[-1]["num_chs"]
        self.fc = nn.Linear(in_channels, fc_dim, bias=False)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        self.l2_normalize = l2_normalize

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_cnn:
            batch_size = x.shape[0]
            x = self.backbone(x)[-1]
            p = self.p if self.training else self.eval_p
            x = gem(x, p).view(batch_size, -1)
        else:
            x = self.backbone(x)
        x = self.fc(x)
        x = self.bn(x)
        if self.l2_normalize:
            x = F.normalize(x)
        return x


class CustomNormalize(nn.Module):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.mean = torch.nn.Parameter(torch.tensor(mean).reshape(1, 3, 1, 1), requires_grad=False)
        self.std = torch.nn.Parameter(torch.tensor(std).reshape(1, 3, 1, 1), requires_grad=False)

    @torch.inference_mode()
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.float()
        tensor = tensor / 255.0
        tensor.sub_(self.mean).div_(self.std)
        return tensor

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class VSCVideoModel(nn.Module):
    def __init__(
        self,
        model,
    ):
        super().__init__()
        self.model = model
        self.mean = model.backbone.default_cfg['mean']
        self.std = model.backbone.default_cfg['std']

    def forward(self, x):
        x = self.model(x)
        return x


class VSCValDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: Path,
        ids: list[str],
        transform: Callable | None = None,
        count: int | None = None,
        fps: int | None = 1,
        temporal_shink_ratio: int = 2,
        len_cap: int | None = None,
        metadata: pd.DataFrame | None = None,
    ):
        self.data_dir = data_dir
        self.ids = ids
        self.transform = transform
        self.count = count
        self.fps = fps
        self.temporal_shink_ratio = temporal_shink_ratio
        self.len_cap = len_cap
        self.metadata = metadata

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        path = self.data_dir / f"{id_}.mp4"

        duration_sec = self.metadata.loc[id_, 'duration_sec']
        if self.len_cap is not None and duration_sec > self.len_cap:
            count = self.len_cap
            fps = None
        else:
            count = self.count
            fps = self.fps

        clip, frame_pos = decord_clip_reader(
            path=path,
            start=0,
            duration=-1,
            count=count,
            fps=fps,
            tensor_format="TCHW",
        )
        if self.transform is not None:
            clip = self.transform(clip)

        video_id = torch.full(
            size=(len(frame_pos) // self.temporal_shink_ratio,),
            fill_value=int(id_[1:]),
            dtype=torch.long,
        )

        return clip, video_id, frame_pos


class VSCLitModule(pl.LightningModule):
    def __init__(
        self,
        model,
        val_transform: Callable | None = None,
        overlap_factor: float = 0.0,
        stack_pred=None,
        stack_pred_thresh=0.01,
        num_views=1,
    ):
        super().__init__()
        self.model = model
        self.val_transform = val_transform
        if self.val_transform is not None:
            self.val_transform = val_transform.to(torch.cuda.current_device())
        self.overlap_factor = overlap_factor
        self.num_views = num_views

        if stack_pred is not None:
            self.maybe_hstack_ids = set(
                stack_pred[stack_pred['hstack'] > stack_pred_thresh]['video_id'].tolist()
            )
            self.maybe_vstack_ids = set(
                stack_pred[stack_pred['vstack'] > stack_pred_thresh]['video_id'].tolist()
            )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0 or dataloader_idx == 2:
            split = "query"
        else:
            split = "reference_or_noise"

        clip, video_ids, frame_pos = batch

        if split == 'query':
            all_crops = []
            all_video_ids = []
            all_frame_pos = []
            for _clip, _video_ids, _frame_pos in zip(clip, video_ids, frame_pos):  # _clip: (T, C, H, W)
                ho = round(_clip.shape[-2] * self.overlap_factor)
                wo = round(_clip.shape[-1] * self.overlap_factor)

                if self.num_views == 1:
                    crops = self.val_transform(_clip)

                elif self.num_views == 2:
                    if f"Q{_video_ids[0].item()}" in self.maybe_hstack_ids:
                        print('maybe hstack:', f"Q{_video_ids[0].item()}")
                        crops = torch.cat([
                            self.val_transform(_clip[:, :, :, :_clip.shape[-1] // 2 + wo]),
                            self.val_transform(_clip[:, :, :, _clip.shape[-1] // 2 - wo:]),
                        ], dim=0)
                        _video_ids = torch.repeat_interleave(_video_ids, repeats=self.num_views)
                        _frame_pos = np.repeat(_frame_pos, self.num_views).reshape(-1, self.num_views).transpose().ravel()
                    elif f"Q{_video_ids[0].item()}" in self.maybe_vstack_ids:
                        print('maybe vstack:', f"Q{_video_ids[0].item()}")
                        crops = torch.cat([
                            self.val_transform(_clip[:, :, :_clip.shape[-2] // 2 + ho, :]),
                            self.val_transform(_clip[:, :, _clip.shape[-2] // 2 - ho:, :]),
                        ], dim=0)
                        _video_ids = torch.repeat_interleave(_video_ids, repeats=self.num_views)
                        _frame_pos = np.repeat(_frame_pos, self.num_views).reshape(-1, self.num_views).transpose().ravel()
                    else:
                        crops = self.val_transform(_clip)

                elif self.num_views == 5:
                    crops = [
                        _clip,
                        _clip[:, :, :_clip.shape[-2] // 2 + ho, :],
                        _clip[:, :, _clip.shape[-2] // 2 - ho:, :],
                        _clip[:, :, :, :_clip.shape[-1] // 2 + wo],
                        _clip[:, :, :, _clip.shape[-1] // 2 - wo:],
                    ]
                    crops = torch.cat([self.val_transform(_) for _ in crops], dim=0)
                    _video_ids = torch.repeat_interleave(_video_ids, repeats=self.num_views)
                    _frame_pos = np.repeat(_frame_pos, self.num_views).reshape(-1, self.num_views).transpose().ravel()

                elif self.num_views == 9:
                    crops = [
                        _clip,
                        _clip[:, :, :_clip.shape[-2] // 2 + ho, :],  # top
                        _clip[:, :, _clip.shape[-2] // 2 - ho:, :],  # bottom
                        _clip[:, :, :, :_clip.shape[-1] // 2 + wo],  # left
                        _clip[:, :, :, _clip.shape[-1] // 2 - wo:],  # right
                        _clip[:, :, :_clip.shape[-2] // 2 + ho, :_clip.shape[-1] // 2 + wo],  # top-left
                        _clip[:, :, :_clip.shape[-2] // 2 + ho, _clip.shape[-1] // 2 - wo:],  # top-right
                        _clip[:, :, _clip.shape[-2] // 2 - ho:, :_clip.shape[-1] // 2 + wo],  # bottom-left
                        _clip[:, :, _clip.shape[-2] // 2 - ho:, _clip.shape[-1] // 2 - wo:],  # bottom-right
                    ]
                    crops = torch.cat([self.val_transform(_) for _ in crops], dim=0)
                    _video_ids = torch.repeat_interleave(_video_ids, repeats=self.num_views)
                    _frame_pos = np.repeat(_frame_pos, self.num_views).reshape(-1, self.num_views).transpose().ravel()

                elif self.num_views == 10:
                    center_margin_ratio = 0.2
                    center_margin_h = round(_clip.shape[-2] * center_margin_ratio)
                    center_margin_w = round(_clip.shape[-1] * center_margin_ratio)
                    crops = [
                        _clip,
                        _clip[:, :, center_margin_h:-center_margin_h, center_margin_w:-center_margin_w],  # center
                        _clip[:, :, :_clip.shape[-2] // 2 + ho, :],  # top
                        _clip[:, :, _clip.shape[-2] // 2 - ho:, :],  # bottom
                        _clip[:, :, :, :_clip.shape[-1] // 2 + wo],  # left
                        _clip[:, :, :, _clip.shape[-1] // 2 - wo:],  # right
                        _clip[:, :, :_clip.shape[-2] // 2 + ho, :_clip.shape[-1] // 2 + wo],  # top-left
                        _clip[:, :, :_clip.shape[-2] // 2 + ho, _clip.shape[-1] // 2 - wo:],  # top-right
                        _clip[:, :, _clip.shape[-2] // 2 - ho:, :_clip.shape[-1] // 2 + wo],  # bottom-left
                        _clip[:, :, _clip.shape[-2] // 2 - ho:, _clip.shape[-1] // 2 - wo:],  # bottom-right
                    ]
                    crops = torch.cat([self.val_transform(_) for _ in crops], dim=0)
                    _video_ids = torch.repeat_interleave(_video_ids, repeats=self.num_views)
                    _frame_pos = np.repeat(_frame_pos, self.num_views).reshape(-1, self.num_views).transpose().ravel()

                elif self.num_views == 12:
                    center_margin_ratio = 0.2
                    center_margin_h = round(_clip.shape[-2] * center_margin_ratio)
                    center_margin_w = round(_clip.shape[-1] * center_margin_ratio)
                    crops = [
                        _clip,
                        F.interpolate(_clip.float(), scale_factor=0.7, mode='bilinear', align_corners=False),
                        F.interpolate(_clip.float(), scale_factor=1.3, mode='bilinear', align_corners=False),
                        _clip[:, :, center_margin_h:-center_margin_h, center_margin_w:-center_margin_w],  # center
                        _clip[:, :, :_clip.shape[-2] // 2 + ho, :],  # top
                        _clip[:, :, _clip.shape[-2] // 2 - ho:, :],  # bottom
                        _clip[:, :, :, :_clip.shape[-1] // 2 + wo],  # left
                        _clip[:, :, :, _clip.shape[-1] // 2 - wo:],  # right
                        _clip[:, :, :_clip.shape[-2] // 2 + ho, :_clip.shape[-1] // 2 + wo],  # top-left
                        _clip[:, :, :_clip.shape[-2] // 2 + ho, _clip.shape[-1] // 2 - wo:],  # top-right
                        _clip[:, :, _clip.shape[-2] // 2 - ho:, :_clip.shape[-1] // 2 + wo],  # bottom-left
                        _clip[:, :, _clip.shape[-2] // 2 - ho:, _clip.shape[-1] // 2 - wo:],  # bottom-right
                    ]
                    crops = torch.cat([self.val_transform(_) for _ in crops], dim=0)
                    _video_ids = torch.repeat_interleave(_video_ids, repeats=self.num_views)
                    _frame_pos = np.repeat(_frame_pos, self.num_views).reshape(-1, self.num_views).transpose().ravel()

                elif self.num_views == 16:
                    crops = [
                        _clip,
                        F.interpolate(_clip.float(), scale_factor=0.7, mode='bilinear', align_corners=False),
                        F.interpolate(_clip.float(), scale_factor=1.3, mode='bilinear', align_corners=False),
                        torchvision.transforms.functional.rotate(_clip, angle=90),
                        torchvision.transforms.functional.rotate(_clip, angle=180),
                        torchvision.transforms.functional.rotate(_clip, angle=270),
                        torchvision.transforms.functional.hflip(_clip),
                        torchvision.transforms.functional.vflip(_clip),
                        _clip[:, :, :_clip.shape[-2] // 2 + ho, :],  # top
                        _clip[:, :, _clip.shape[-2] // 2 - ho:, :],  # bottom
                        _clip[:, :, :, :_clip.shape[-1] // 2 + wo],  # left
                        _clip[:, :, :, _clip.shape[-1] // 2 - wo:],  # right
                        _clip[:, :, :_clip.shape[-2] // 2 + ho, :_clip.shape[-1] // 2 + wo],  # top-left
                        _clip[:, :, :_clip.shape[-2] // 2 + ho, _clip.shape[-1] // 2 - wo:],  # top-right
                        _clip[:, :, _clip.shape[-2] // 2 - ho:, :_clip.shape[-1] // 2 + wo],  # bottom-left
                        _clip[:, :, _clip.shape[-2] // 2 - ho:, _clip.shape[-1] // 2 - wo:],  # bottom-right
                    ]
                    crops = torch.cat([self.val_transform(_) for _ in crops], dim=0)
                    _video_ids = torch.repeat_interleave(_video_ids, repeats=self.num_views)
                    _frame_pos = np.repeat(_frame_pos, self.num_views).reshape(-1, self.num_views).transpose().ravel()

                elif self.num_views == 30:
                    center_margin_ratio = 0.2
                    center_margin_h = round(_clip.shape[-2] * center_margin_ratio)
                    center_margin_w = round(_clip.shape[-1] * center_margin_ratio)

                    crops = []
                    for _clip_t in [
                        _clip,
                        torchvision.transforms.functional.rotate(_clip, angle=90),
                        torchvision.transforms.functional.rotate(_clip, angle=180),
                        torchvision.transforms.functional.rotate(_clip, angle=270),
                        torchvision.transforms.functional.hflip(_clip),
                        torchvision.transforms.functional.vflip(_clip),
                    ]:
                        _crops = [
                            _clip_t,
                            # _clip_t[:, :, center_margin_h:-center_margin_h, center_margin_w:-center_margin_w],  # center
                            _clip_t[:, :, :_clip_t.shape[-2] // 2 + ho, :],  # top
                            _clip_t[:, :, _clip_t.shape[-2] // 2 - ho:, :],  # bottom
                            _clip_t[:, :, :, :_clip_t.shape[-1] // 2 + wo],  # left
                            _clip_t[:, :, :, _clip_t.shape[-1] // 2 - wo:],  # right
                            # _clip_t[:, :, :_clip_t.shape[-2] // 2 + ho, :_clip_t.shape[-1] // 2 + wo],  # top-left
                            # _clip_t[:, :, :_clip_t.shape[-2] // 2 + ho, _clip_t.shape[-1] // 2 - wo:],  # top-right
                            # _clip_t[:, :, _clip_t.shape[-2] // 2 - ho:, :_clip_t.shape[-1] // 2 + wo],  # bottom-left
                            # _clip_t[:, :, _clip_t.shape[-2] // 2 - ho:, _clip_t.shape[-1] // 2 - wo:],  # bottom-right
                        ]
                        crops.extend(_crops)
                    crops = torch.cat([self.val_transform(_) for _ in crops], dim=0)
                    _video_ids = torch.repeat_interleave(_video_ids, repeats=self.num_views)
                    _frame_pos = np.repeat(_frame_pos, self.num_views).reshape(-1, self.num_views).transpose().ravel()

                all_crops.append(crops)
                all_video_ids.append(_video_ids)
                all_frame_pos.append(_frame_pos)

            clip = torch.cat(all_crops, dim=0)
            video_ids = torch.cat(all_video_ids)
            frame_pos = np.concatenate(all_frame_pos)
        else:
            clip = torch.cat([self.val_transform(_) for _ in clip], dim=0)
            video_ids = torch.cat(video_ids)
            frame_pos = np.concatenate(frame_pos)

        clip = clip.half()
        embeddings = self.model(clip)
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        if batch_idx < 10:
            print(embeddings.shape, video_ids.shape, frame_pos.shape)

        return embeddings, video_ids, frame_pos


def predict(args, pl_model=None):
    # cudnn can be problematic
    torch.backends.cudnn.enabled = False

    pred_output_dir = Path(args.pred_output_dir)

    if args.dryrun:
        sample_size = 32
    else:
        sample_size = 9999999

    stack_pred = pd.read_csv(args.copy_type_pred_path)
    print('stack_pred', len(stack_pred))

    query_metadata = pd.read_csv(args.query_metadata_path, usecols=["video_id", "duration_sec"])
    ref_metadata = pd.read_csv(args.ref_metadata_path, usecols=["video_id", "duration_sec"])
    test_query_metadata = pd.read_csv(args.test_query_metadata_path, usecols=["video_id", "duration_sec"])
    test_ref_metadata = pd.read_csv(args.test_ref_metadata_path, usecols=["video_id", "duration_sec"])
    metadata = pd.concat([query_metadata, ref_metadata, test_query_metadata, test_ref_metadata], axis=0).set_index("video_id")

    query_ids = query_metadata['video_id'].tolist()[:sample_size]
    ref_ids = ref_metadata['video_id'].tolist()[:sample_size]
    test_query_ids = test_query_metadata['video_id'].tolist()[:sample_size]
    test_ref_ids = test_ref_metadata['video_id'].tolist()[:sample_size]

    input_size = tuple(map(int, args.input_size.split("x")))
    if len(input_size) == 1:
        input_size = (input_size[0], input_size[0])

    if args.arch in ["isc_selfsup_v98", "isc_ft_v107"]:
        model, preprocessor = create_model(
            weight_name=args.arch,
            device="cpu",
            fc_dim=args.feature_dim,
            p=args.gem_p,
            eval_p=args.gem_eval_p,
            l2_normalize=args.l2_normalize,
            is_training=False,
        )
        backbone = model.backbone
    else:
        try:
            backbone = timm.create_model(args.arch, features_only=True, pretrained=True)
        except:
            backbone = timm.create_model(args.arch, pretrained=True, num_classes=0, img_size=input_size)
        model = ISCNet(
            backbone=backbone,
            fc_dim=args.feature_dim,
            p=args.gem_p,
            eval_p=args.gem_eval_p,
            l2_normalize=args.l2_normalize,
        )
    model = VSCVideoModel(model=model)

    normalize_rgb = CustomNormalize(mean=model.mean, std=model.std)
    t_val = nn.Sequential(
        Resize(size=input_size),
        normalize_rgb,
    )
    t_val = torch.jit.script(t_val)

    pred_datasets = [
        VSCValDataset(
            data_dir=Path(args.query_video_dir),
            ids=query_ids,
            fps=args.fps,
            temporal_shink_ratio=args.temporal_shink_ratio,
            metadata=metadata,
            len_cap=args.len_cap,
        ),
    ]
    if not args.only_train_query:
        pred_datasets.extend([
            VSCValDataset(
                data_dir=Path(args.ref_video_dir),
                ids=ref_ids,
                fps=args.fps,
                temporal_shink_ratio=args.temporal_shink_ratio,
                metadata=metadata,
                len_cap=args.len_cap,
            ),
            VSCValDataset(
                data_dir=Path(args.test_query_video_dir),
                ids=test_query_ids,
                fps=args.fps,
                temporal_shink_ratio=args.temporal_shink_ratio,
                metadata=metadata,
                len_cap=args.len_cap,
            ),
            VSCValDataset(
                data_dir=Path(args.test_ref_video_dir),
                ids=test_ref_ids,
                fps=args.fps,
                temporal_shink_ratio=args.temporal_shink_ratio,
                metadata=metadata,
                len_cap=args.len_cap,
            ),
        ])

    pl_dm = VSCLitDataModule(
        train_datasets=[
            # dummy
            VSCValDataset(
                data_dir=Path(args.query_video_dir),
                ids=query_ids,
                fps=args.fps,
                temporal_shink_ratio=args.temporal_shink_ratio,
                metadata=metadata,
            )
        ],
        val_datasets=pred_datasets,
        train_batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        workers=args.workers,
        collate_fn=collate_fn_for_variable_size,
    )

    if pl_model is None:
        if args.weight is not None:
            state_dict = torch.load(args.weight, map_location="cpu")

            if not list(state_dict.keys())[0].startswith("model."):
                new_state_dict = {}
                for key in state_dict.keys():
                    new_state_dict["model." + key] = state_dict[key]
                state_dict = new_state_dict

            try:
                model.load_state_dict(state_dict)
            except RuntimeError as e:
                print(e)
                print("load state_dict with strict=False")
                model.load_state_dict(state_dict, strict=False)

        pl_model = VSCLitModule(
            model=model,
            val_transform=t_val,
            overlap_factor=args.overlap_factor,
            stack_pred=stack_pred,
            stack_pred_thresh=args.stack_pred_thresh,
            num_views=args.num_views,
        )

    pred_writer = CustomWriter(output_dir=pred_output_dir, write_interval="epoch")
    trainer = pl.Trainer(
        devices=-1,
        strategy=args.ddp_strategy,
        accelerator="gpu",
        precision=16,
        num_sanity_val_steps=0,
        callbacks=[pred_writer],
    )
    trainer.predict(pl_model, dataloaders=pl_dm.val_dataloader(), return_predictions=False)

    torch.distributed.barrier()

    if trainer.global_rank == 0:
        if args.gt_path is None:
            gt_matches = None
        else:
            gt_matches = Match.read_csv(args.gt_path, is_gt=True)
        aggregate_preds_and_evaluate(pred_output_dir, gt_matches, only_train_query=args.only_train_query)


def aggregate_preds_and_evaluate(pred_output_dir, gt_matches=None, only_train_query=False):
    batch_indices_paths = sorted(pred_output_dir.glob("batch_indices*"))
    pred_paths = sorted(pred_output_dir.glob("predictions*"))

    split_names = ["query"] if only_train_query else ["query", "ref", "test_query", "test_ref"]
    video_features = {split_name: [] for split_name in split_names}

    for batch_indices_path, pred_path in zip(batch_indices_paths, pred_paths):
        predictions = torch.load(pred_path)
        for i_split, split in enumerate(split_names):
            embeddings_list, video_ids_list, frame_pos_list = list(zip(*predictions[i_split]))
            for embeddings, video_ids, frame_pos in zip(embeddings_list, video_ids_list, frame_pos_list):
                for video_id, start, end in same_value_ranges(video_ids):
                    prefix = 'Q' if 'query' in split else 'R'
                    video_features[split].append(
                        VideoFeature(
                            video_id=prefix + str(video_id.item()),
                            timestamps=frame_pos[start:end].astype(np.float32),
                            feature=embeddings[start:end].numpy(),
                        )
                    )
        os.remove(batch_indices_path)
        os.remove(pred_path)

    for split, vfs in video_features.items():
        store_features(pred_output_dir / f"{split}.npz", vfs)

    if gt_matches is None:
        return

    queries, refs = score_normalize(
        video_features["query"],
        video_features["ref"],
        video_features["test_ref"],
        beta=1.2,
    )

    print("queries:", sum(len(vf.feature) for vf in queries))
    candidates = knn_search(queries, refs, retrieve_per_query=25, use_gpu=True)
    CandidatePair.write_csv(candidates, pred_output_dir / "candidates.csv")
    print("candidates saved:", pred_output_dir / "candidates.csv")

    ap = average_precision(CandidatePair.from_matches(gt_matches), candidates)
    logger.info(f"AP: {ap.ap:.4f}")
    return ap


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VSC Image Level Training")
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
        "--test_query_video_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--test_ref_video_dir",
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
        "--test_query_metadata_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--test_ref_metadata_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        # required=True,
    )
    parser.add_argument("-a", "--arch", metavar="ARCH", default="swin_base_patch4_window7_224")
    parser.add_argument("--workers", default=os.cpu_count(), type=int)
    parser.add_argument("-b", "--batch-size", default=512, type=int)
    parser.add_argument("--val-batch-size", default=1024, type=int)
    parser.add_argument("--gem-p", default=1.0, type=float)
    parser.add_argument("--gem-eval-p", default=1.0, type=float)
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--input-size", default=224)
    parser.add_argument("--weight", type=str)
    parser.add_argument("--feature_dim", default=512, type=int)
    parser.add_argument("--l2_normalize", type=bool, default=True)
    parser.add_argument("--ddp_strategy", default="deepspeed_stage2", type=str)
    parser.add_argument("--num_frames", type=int)
    parser.add_argument("--temporal_shink_ratio", default=1, type=int)
    parser.add_argument("--pred_output_dir", type=str, default=f"{ver}/preds")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--num_views", default=1, type=int)
    parser.add_argument("--overlap_factor", default=0.0, type=float)
    parser.add_argument("--copy_type_pred_path", type=str, required=True)
    parser.add_argument("--stack_pred_thresh", default=0.15, type=float)
    parser.add_argument("--len_cap", type=int)
    parser.add_argument("--only_train_query", action="store_true")
    args = parser.parse_args()

    predict(args)
