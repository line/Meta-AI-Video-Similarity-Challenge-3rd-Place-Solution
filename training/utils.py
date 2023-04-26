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

import collections
import dataclasses
import json
import math
import pickle
import random
import shutil
from pathlib import Path
from typing import Any, Collection, Dict, List, Optional, TextIO, Tuple, Union
from weakref import proxy

import decord
import faiss
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from augly.image.functional import overlay_text
from augly.image.transforms import BaseTransform
from augly.utils.base_paths import MODULE_BASE_DIR
from augly.utils.constants import FONT_LIST_PATH, FONTS_DIR
from loguru import logger
from PIL import Image, ImageFilter
from pytorch_lightning.callbacks import BasePredictionWriter, ModelCheckpoint
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from vsc.candidates import ScoreAggregation
from vsc.index import PairMatches, VideoFeature, VideoIndex
from vsc.metrics import (
    CandidatePair,
    Dataset,
    Match,
    format_video_id,
)


class NCropsTransform:
    """Take n random crops of one image as the query and key."""

    def __init__(self, aug_moderate, aug_hard, ncrops=2):
        self.aug_moderate = aug_moderate
        self.aug_hard = aug_hard
        self.ncrops = ncrops

    def __call__(self, x):
        return [self.aug_moderate(x)] + [
            self.aug_hard(x) for _ in range(self.ncrops - 1)
        ]


class RandomOverlayText(BaseTransform):
    def __init__(
        self,
        opacity: float = 1.0,
        p: float = 1.0,
    ):
        super().__init__(p)
        self.opacity = opacity

        with open(Path(FONTS_DIR) / FONT_LIST_PATH) as f:
            font_list = [s.strip() for s in f.readlines()]
            blacklist = [
                "TypeMyMusic",
                "PainttheSky-Regular",
            ]
            self.font_list = [
                f for f in font_list if all(_ not in f for _ in blacklist)
            ]

        self.font_lens = []
        for ff in self.font_list:
            font_file = Path(MODULE_BASE_DIR) / ff.replace(".ttf", ".pkl")
            with open(font_file, "rb") as f:
                self.font_lens.append(len(pickle.load(f)))

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        i = random.randrange(0, len(self.font_list))
        kwargs = dict(
            font_file=Path(MODULE_BASE_DIR) / self.font_list[i],
            font_size=random.uniform(0.1, 0.3),
            color=[random.randrange(0, 256) for _ in range(3)],
            x_pos=random.uniform(0.0, 0.5),
            metadata=metadata,
            opacity=self.opacity,
        )
        try:
            for j in range(random.randrange(1, 3)):
                if j == 0:
                    y_pos = random.uniform(0.0, 0.5)
                else:
                    y_pos += kwargs["font_size"]
                image = overlay_text(
                    image,
                    text=[
                        random.randrange(0, self.font_lens[i])
                        for _ in range(random.randrange(5, 10))
                    ],
                    y_pos=y_pos,
                    **kwargs,
                )
            return image
        except OSError:
            return image


class RandomEdgeEnhance(BaseTransform):
    def __init__(
        self,
        mode=ImageFilter.EDGE_ENHANCE,
        p: float = 1.0,
    ):
        super().__init__(p)
        self.mode = mode

    def apply_transform(self, image: Image.Image, *args) -> Image.Image:
        return image.filter(self.mode)


class ShuffledAug:
    def __init__(self, aug_list):
        self.aug_list = aug_list

    def __call__(self, x):
        # without replacement
        shuffled_aug_list = random.sample(self.aug_list, len(self.aug_list))
        for op in shuffled_aug_list:
            x = op(x)
        return x


def convert2rgb(x):
    return x.convert("RGB")


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * (1 - (epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if "fix_lr" in param_group and param_group["fix_lr"]:
            param_group["lr"] = init_lr
        else:
            param_group["lr"] = cur_lr


def cosine_scheduler(optimizer, training_steps, warmup_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = current_step - warmup_steps
        progress /= max(1, training_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def constant_scheduler(optimizer, training_steps, warmup_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def collate_fn_for_variable_size(batch):
    return list(zip(*batch))


class VSCLitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_datasets,
        val_datasets,
        train_batch_size: int = 8,
        val_batch_size: int = 16,
        workers: int = 4,
        collate_fn=None,
    ):
        super().__init__()
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.workers = workers
        self.collate_fn = collate_fn

    def train_dataloader(self):
        if len(self.train_datasets) == 1:  # for back compatibility
            return DataLoader(
                self.train_datasets[0],
                batch_size=self.train_batch_size,
                num_workers=self.workers,
                pin_memory=True,
                shuffle=True,
                drop_last=True,
                collate_fn=self.collate_fn,
            )
        else:
            return [
                DataLoader(
                    dataset,
                    batch_size=self.train_batch_size,
                    num_workers=self.workers,
                    pin_memory=True,
                    shuffle=True,
                    drop_last=True,
                    collate_fn=self.collate_fn,
                )
                for dataset in self.train_datasets
            ]

    def val_dataloader(self):
        return [
            DataLoader(
                dataset,
                batch_size=self.val_batch_size,
                num_workers=self.workers,
                pin_memory=True,
                shuffle=False,
                drop_last=False,
                collate_fn=self.collate_fn,
            )
            for dataset in self.val_datasets
        ]


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        save_dir = Path(filepath)
        save_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(trainer.model, "module"):
            args = trainer.model.module._forward_module.args
            model = trainer.model.module._forward_module.model
        else:
            args = trainer.model.args
            model = trainer.model.model

        with open(save_dir / "args.json", "w") as f:
            for k, v in vars(args).items():
                if isinstance(v, Path):
                    vars(args)[k] = str(v)
            json.dump(vars(args), f, indent=4)

        torch.save(model.state_dict(), save_dir / "model.pth")

        self._last_global_step_saved = trainer.global_step

        # notify loggers
        if trainer.is_global_zero:
            print(f"Saving model checkpoint to {save_dir}")
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(
            predictions, self.output_dir / f"predictions_{trainer.global_rank}.pth"
        )

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        torch.save(
            batch_indices, self.output_dir / f"batch_indices_{trainer.global_rank}.pth"
        )


def decord_clip_reader(
    path: Path | str,
    start: float = 0.0,
    duration: float | None = None,
    count: int | None = None,
    fps: float | None = None,
    tensor_format: str = "TCHW",
    randomize_frame_pos: bool = False,
):
    assert count is None or fps is None, "Either count or fps must be specified"

    num_threads = 0
    ctx = decord.cpu(0)

    reader = decord.VideoReader(
        Path(path).as_posix(),
        ctx=ctx,
        num_threads=num_threads,
    )

    n_frames = len(reader)
    timestamps = reader.get_frame_timestamp(np.arange(n_frames))
    end_timestamps = timestamps[:, 1]
    clip_duration_sec = end_timestamps[-1]
    clip_fps = n_frames / clip_duration_sec

    if count is None and fps is None:
        # return all frames
        frame_pos = end_timestamps
        frame_ids = np.arange(n_frames)

    else:
        if duration is None or duration <= 0:
            duration = clip_duration_sec - start

        if count is None and fps is not None:
            if fps > clip_fps:
                fps = clip_fps
            count = max(1, int(duration * fps))

        if fps is None:
            step_size = duration / count
        else:
            step_size = 1 / fps
        frame_pos = np.array([(i * step_size) + start for i in range(count)])
        if randomize_frame_pos:
            frame_pos = frame_pos + random.uniform(0, step_size)
        else:
            frame_pos += step_size / 2  # to match ffmpeg behavior
        frame_ids = np.searchsorted(end_timestamps, frame_pos)
        frame_ids = np.minimum(frame_ids, n_frames - 1)

    frames = reader.get_batch(frame_ids).asnumpy()

    tensor = torch.as_tensor(frames)
    if tensor_format == "TCHW":
        tensor = tensor.permute(0, 3, 1, 2)
    elif tensor_format == "CTHW":
        tensor = tensor.permute(3, 0, 1, 2)
    return tensor, frame_pos


class CustomNormalize(nn.Module):
    def __init__(
        self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), format="TCHW"
    ):
        super().__init__()
        if format == "TCHW":
            self.mean = torch.nn.Parameter(
                torch.tensor(mean).reshape(1, 3, 1, 1), requires_grad=False
            )
            self.std = torch.nn.Parameter(
                torch.tensor(std).reshape(1, 3, 1, 1), requires_grad=False
            )
        elif format == "CTHW":
            self.mean = torch.nn.Parameter(
                torch.tensor(mean).reshape(3, 1, 1, 1), requires_grad=False
            )
            self.std = torch.nn.Parameter(
                torch.tensor(std).reshape(3, 1, 1, 1), requires_grad=False
            )
        else:
            raise ValueError(f"Unknown format: {format}")

    @torch.inference_mode()
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.float()
        tensor = tensor / 255.0
        tensor.sub_(self.mean).div_(self.std)
        return tensor

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class CustomVideoIndex(VideoIndex):
    def _knn_search(self, query_features: np.ndarray, k):
        index = self.index
        if self.use_gpu and faiss.get_num_gpus() > 0:
            logger.info("Moving index to GPU")
            index = faiss.index_cpu_to_all_gpus(self.index)
        logger.info("Performing KNN search")
        similarity, ids = index.search(query_features, k)
        for i in range(ids.shape[0]):
            for j in range(ids.shape[1]):
                yield (i, ids[i, j], similarity[i, j])


@dataclasses.dataclass
class CustomCandidatePair(CandidatePair):
    query_max_idx: float | None = None
    ref_max_idx: float | None = None

    @classmethod
    def to_dataframe(
        cls,
        candidates: Collection["CustomCandidatePair"],
        do_format_video_id: bool = True,
    ) -> pd.DataFrame:
        rows = []
        for c in candidates:
            row = {
                "query_id": format_video_id(c.query_id, Dataset.QUERIES)
                if do_format_video_id
                else c.query_id,
                "ref_id": format_video_id(c.ref_id, Dataset.REFS)
                if do_format_video_id
                else c.ref_id,
                "score": c.score,
            }
            if c.query_max_idx is not None:
                row["query_max_idx"] = c.query_max_idx
            if c.ref_max_idx is not None:
                row["ref_max_idx"] = c.ref_max_idx
            rows.append(row)
        return pd.DataFrame(rows)

    @classmethod
    def write_csv(
        cls,
        candidates: Collection["CustomCandidatePair"],
        file: Union[str, TextIO],
        do_format_video_id: bool = True,
    ):
        df = cls.to_dataframe(candidates, do_format_video_id)
        df.to_csv(file, index=False)

    @classmethod
    def read_csv(
        cls, file: Union[str, TextIO], do_format_video_id: bool = True
    ) -> List["CustomCandidatePair"]:
        df = pd.read_csv(file)
        pairs = []
        for _, row in df.iterrows():
            query_id = (
                format_video_id(row.query_id, Dataset.QUERIES)
                if do_format_video_id
                else row.query_id
            )
            ref_id = (
                format_video_id(row.ref_id, Dataset.REFS)
                if do_format_video_id
                else row.ref_id
            )
            pairs.append(
                CustomCandidatePair(query_id=query_id, ref_id=ref_id, score=row.score)
            )
        return pairs

    @classmethod
    def from_matches(cls, matches: Collection["Match"]) -> List["CustomCandidatePair"]:
        scores = collections.defaultdict(float)
        for match in matches:
            key = (match.query_id, match.ref_id)
            scores[key] = max(match.score, scores[key])
        return [
            CustomCandidatePair(query_id=query_id, ref_id=ref_id, score=score)
            for ((query_id, ref_id), score) in scores.items()
        ]


class MaxScoreAggregationWithTimestamp(ScoreAggregation):
    def aggregate(self, match: PairMatches) -> float:
        scores = [m.score for m in match.matches]
        argmax = np.argmax(scores)
        return (
            match.matches[argmax].score,
            match.matches[argmax].query_timestamps,
            match.matches[argmax].ref_timestamps,
        )

    def score(self, match: PairMatches) -> CustomCandidatePair:
        score, query_max_idx, ref_max_idx = self.aggregate(match)
        return CustomCandidatePair(
            query_id=match.query_id,
            ref_id=match.ref_id,
            score=score,
            query_max_idx=query_max_idx[0],
            ref_max_idx=ref_max_idx[0],
        )


class CustomCandidateGeneration:
    def __init__(self, references: list[VideoFeature], aggregation: ScoreAggregation):
        self.aggregation = aggregation
        dim = references[0].dimensions()
        self.index = CustomVideoIndex(dim)
        self.index.add(references)

    def query(
        self, queries: list[VideoFeature], global_k: int
    ) -> list[CustomCandidatePair]:
        matches = self.index.search(queries, global_k=global_k)
        candidates = [self.aggregation.score(match) for match in matches]
        candidates = sorted(candidates, key=lambda match: match.score, reverse=True)
        return candidates


def knn_search(
    queries: list[VideoFeature],
    refs: list[VideoFeature],
    retrieve_per_query: int = 25,
    use_gpu: bool = True,
) -> list[CandidatePair]:
    aggregation = MaxScoreAggregationWithTimestamp()
    logger.info("Searching")
    cg = CustomCandidateGeneration(refs, aggregation)
    cg.index.use_gpu = use_gpu
    candidates = cg.query(queries, global_k=-retrieve_per_query)
    logger.info(f"Got {len(candidates)} candidates")
    return candidates


class TTA30ViewsTransform(nn.Module):
    def __init__(self, base_transforms=None):
        super().__init__()
        self.base_transforms = base_transforms

    def forward(self, x) -> torch.Tensor:
        *_, h, w = x.shape

        x_top = x[..., : h // 2, :]  # top
        x_bottom = x[..., h // 2 :, :]  # bottom
        x_left = x[..., :, : w // 2]  # left
        x_right = x[..., :, w // 2 :]  # right

        if self.base_transforms is not None:
            x = self.base_transforms(x)
            x_top = self.base_transforms(x_top)
            x_bottom = self.base_transforms(x_bottom)
            x_left = self.base_transforms(x_left)
            x_right = self.base_transforms(x_right)

        crops = [
            x,
            torchvision.transforms.functional.rotate(x, angle=90),
            torchvision.transforms.functional.rotate(x, angle=180),
            torchvision.transforms.functional.rotate(x, angle=270),
            torchvision.transforms.functional.hflip(x),
            torchvision.transforms.functional.vflip(x),
            x_top,
            torchvision.transforms.functional.rotate(x_top, angle=90),
            torchvision.transforms.functional.rotate(x_top, angle=180),
            torchvision.transforms.functional.rotate(x_top, angle=270),
            torchvision.transforms.functional.hflip(x_top),
            torchvision.transforms.functional.vflip(x_top),
            x_bottom,
            torchvision.transforms.functional.rotate(x_bottom, angle=90),
            torchvision.transforms.functional.rotate(x_bottom, angle=180),
            torchvision.transforms.functional.rotate(x_bottom, angle=270),
            torchvision.transforms.functional.hflip(x_bottom),
            torchvision.transforms.functional.vflip(x_bottom),
            x_left,
            torchvision.transforms.functional.rotate(x_left, angle=90),
            torchvision.transforms.functional.rotate(x_left, angle=180),
            torchvision.transforms.functional.rotate(x_left, angle=270),
            torchvision.transforms.functional.hflip(x_left),
            torchvision.transforms.functional.vflip(x_left),
            x_right,
            torchvision.transforms.functional.rotate(x_right, angle=90),
            torchvision.transforms.functional.rotate(x_right, angle=180),
            torchvision.transforms.functional.rotate(x_right, angle=270),
            torchvision.transforms.functional.hflip(x_right),
            torchvision.transforms.functional.vflip(x_right),
        ]
        crops = torch.cat(crops, dim=0)

        return crops


class TTA24ViewsTransform(nn.Module):
    def __init__(self, base_transforms=None):
        super().__init__()
        self.base_transforms = base_transforms

    def forward(self, x) -> torch.Tensor:
        *_, h, w = x.shape

        x_top_left = x[..., : h // 2, : w // 2]  # top_left
        x_top_right = x[..., : h // 2, w // 2 :]  # top_right
        x_bottom_left = x[..., h // 2 :, : w // 2]  # bottom_left
        x_bottom_right = x[..., h // 2 :, w // 2 :]  # bottom_right

        if self.base_transforms is not None:
            x_top_left = self.base_transforms(x_top_left)
            x_top_right = self.base_transforms(x_top_right)
            x_bottom_left = self.base_transforms(x_bottom_left)
            x_bottom_right = self.base_transforms(x_bottom_right)

        crops = [
            x_top_left,
            torchvision.transforms.functional.rotate(x_top_left, angle=90),
            torchvision.transforms.functional.rotate(x_top_left, angle=180),
            torchvision.transforms.functional.rotate(x_top_left, angle=270),
            torchvision.transforms.functional.hflip(x_top_left),
            torchvision.transforms.functional.vflip(x_top_left),
            x_top_right,
            torchvision.transforms.functional.rotate(x_top_right, angle=90),
            torchvision.transforms.functional.rotate(x_top_right, angle=180),
            torchvision.transforms.functional.rotate(x_top_right, angle=270),
            torchvision.transforms.functional.hflip(x_top_right),
            torchvision.transforms.functional.vflip(x_top_right),
            x_bottom_left,
            torchvision.transforms.functional.rotate(x_bottom_left, angle=90),
            torchvision.transforms.functional.rotate(x_bottom_left, angle=180),
            torchvision.transforms.functional.rotate(x_bottom_left, angle=270),
            torchvision.transforms.functional.hflip(x_bottom_left),
            torchvision.transforms.functional.vflip(x_bottom_left),
            x_bottom_right,
            torchvision.transforms.functional.rotate(x_bottom_right, angle=90),
            torchvision.transforms.functional.rotate(x_bottom_right, angle=180),
            torchvision.transforms.functional.rotate(x_bottom_right, angle=270),
            torchvision.transforms.functional.hflip(x_bottom_right),
            torchvision.transforms.functional.vflip(x_bottom_right),
        ]
        crops = torch.cat(crops, dim=0)

        return crops
