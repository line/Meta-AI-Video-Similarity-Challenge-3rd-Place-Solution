from __future__ import annotations
import collections

import dataclasses
import json
import math
import pickle
import random
import shutil
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple, Collection, Union
from weakref import proxy

import decord
import faiss
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import Parameter
from augly.image.functional import overlay_image, overlay_text
from augly.image.transforms import BaseTransform
from augly.utils.base_paths import MODULE_BASE_DIR
from augly.utils.constants import FONT_LIST_PATH, FONTS_DIR
from PIL import Image, ImageFilter
from pytorch_lightning.callbacks import ModelCheckpoint, BasePredictionWriter
from sklearn.preprocessing import normalize
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from loguru import logger

from vsc.baseline.score_normalization import transform_features
from vsc.candidates import CandidateGeneration, MaxScoreAggregation, ScoreAggregation
from vsc.index import PairMatches, VideoFeature, VideoIndex
from vsc.metrics import CandidatePair, Dataset, Match, average_precision, format_video_id


class NCropsTransform:
    """Take n random crops of one image as the query and key."""

    def __init__(self, aug_moderate, aug_hard, ncrops=2):
        self.aug_moderate = aug_moderate
        self.aug_hard = aug_hard
        self.ncrops = ncrops

    def __call__(self, x):
        return [self.aug_moderate(x)] + [self.aug_hard(x) for _ in range(self.ncrops - 1)]


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
            self.font_list = [f for f in font_list if all(_ not in f for _ in blacklist)]

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
        collate_fn = None,
    ):
        super().__init__()
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.workers = workers
        self.collate_fn = collate_fn

    def train_dataloader(self):
        if len(self.train_datasets) == 1:  # 後方互換性のため
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


def negative_embedding_subtraction(
    queries: List[VideoFeature],
    refs: List[VideoFeature],
    score_norm_refs: List[VideoFeature],
    pre_l2_normalize: bool = False,
    post_l2_normalize: bool = False,
    beta: float = 1.0,
    k: int = 10,
    alpha: float = 1.0,
) -> Tuple[List[VideoFeature], List[VideoFeature]]:
    # impl of https://arxiv.org/abs/2112.04323

    if pre_l2_normalize:
        logger.info("L2 normalizing")
        queries, refs, score_norm_refs = [
            transform_features(x, normalize) for x in [queries, refs, score_norm_refs]
        ]

    logger.info("Applying negative embedding subtraction")
    index = CandidateGeneration(score_norm_refs, MaxScoreAggregation()).index.index
    if faiss.get_num_gpus() > 0:
        index = faiss.index_cpu_to_all_gpus(index)

    negative_embeddings = np.concatenate([vf.feature for vf in score_norm_refs], axis=0)

    adapted_queries = []
    for query in queries:
        similarity, ids = index.search(query.feature, k=k)
        weights = similarity[..., None] ** alpha
        topk_negative_embeddings = negative_embeddings[ids] * weights
        subtracted_embedding = topk_negative_embeddings.mean(axis=1) * beta
        adapted_embedding = query.feature - subtracted_embedding
        adapted_queries.append(dataclasses.replace(query, feature=adapted_embedding))

    adapted_refs = []
    for ref in refs:
        similarity, ids = index.search(ref.feature, k=k)
        weights = similarity[..., None] ** alpha
        topk_negative_embeddings = negative_embeddings[ids] * weights
        subtracted_embedding = topk_negative_embeddings.mean(axis=1) * beta
        adapted_embedding = ref.feature - subtracted_embedding
        adapted_refs.append(dataclasses.replace(ref, feature=adapted_embedding))

    if post_l2_normalize:
        logger.info("L2 normalizing")
        adapted_queries, adapted_refs = [
            transform_features(x, normalize) for x in [adapted_queries, adapted_refs]
        ]

    return adapted_queries, adapted_refs


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(predictions, self.output_dir / f"predictions_{trainer.global_rank}.pth")

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        torch.save(batch_indices, self.output_dir / f"batch_indices_{trainer.global_rank}.pth")


def decord_clip_reader(
    path: Path | str,
    start: float = 0.,
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
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), format="TCHW"):
        super().__init__()
        if format == "TCHW":
            self.mean = torch.nn.Parameter(torch.tensor(mean).reshape(1, 3, 1, 1), requires_grad=False)
            self.std = torch.nn.Parameter(torch.tensor(std).reshape(1, 3, 1, 1), requires_grad=False)
        elif format == "CTHW":
            self.mean = torch.nn.Parameter(torch.tensor(mean).reshape(3, 1, 1, 1), requires_grad=False)
            self.std = torch.nn.Parameter(torch.tensor(std).reshape(3, 1, 1, 1), requires_grad=False)
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


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.kernel = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.kernel)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.kernel))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output, cosine


class SubCenterArcFace(nn.Module):
    """Implementation of
    `Sub-center ArcFace: Boosting Face Recognition
    by Large-scale Noisy Web Faces`_.
    .. _Sub-center ArcFace\: Boosting Face Recognition \
        by Large-scale Noisy Web Faces:
        https://ibug.doc.ic.ac.uk/media/uploads/documents/eccv_1445.pdf
    Args:
        in_features: size of each input sample.
        out_features: size of each output sample.
        s: norm of input feature,
            Default: ``64.0``.
        m: margin.
            Default: ``0.5``.
        k: number of possible class centroids.
            Default: ``3``.
        eps (float, optional): operation accuracy.
            Default: ``1e-6``.
    Shape:
        - Input: :math:`(batch, H_{in})` where
          :math:`H_{in} = in\_features`.
        - Output: :math:`(batch, H_{out})` where
          :math:`H_{out} = out\_features`.
    Example:
        >>> layer = SubCenterArcFace(5, 10, s=1.31, m=0.35, k=2)
        >>> loss_fn = nn.CrosEntropyLoss()
        >>> embedding = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(10)
        >>> output = layer(embedding, target)
        >>> loss = loss_fn(output, target)
        >>> loss.backward()
    """

    def __init__(  # noqa: D107
        self,
        in_features: int,
        out_features: int,
        s: float = 64.0,
        m: float = 0.5,
        k: int = 2,
        eps: float = 1e-6,
    ):
        super(SubCenterArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.s = s
        self.m = m
        self.k = k
        self.eps = eps

        self.kernel = nn.Parameter(torch.FloatTensor(k, in_features, out_features))
        nn.init.xavier_uniform_(self.kernel)

        self.threshold = math.pi - self.m

    def __repr__(self) -> str:
        """Object representation."""
        rep = (
            "SubCenterArcFace("
            f"in_features={self.in_features},"
            f"out_features={self.out_features},"
            f"s={self.s},"
            f"m={self.m},"
            f"k={self.k},"
            f"eps={self.eps}"
            ")"
        )
        return rep

    def forward(self, input: torch.Tensor, target: torch.LongTensor = None) -> torch.Tensor:
        """
        Args:
            input: input features,
                expected shapes ``BxF`` where ``B``
                is batch dimension and ``F`` is an
                input feature dimension.
            target: target classes,
                expected shapes ``B`` where
                ``B`` is batch dimension.
                If `None` then will be returned
                projection on centroids.
                Default is `None`.
        Returns:
            tensor (logits) with shapes ``BxC``
            where ``C`` is a number of classes.
        """
        feats = F.normalize(input).unsqueeze(0).expand(self.k, *input.shape)  # k*b*f
        wght = F.normalize(self.kernel, dim=1)  # k*f*c
        cos_theta = torch.bmm(feats, wght)  # k*b*f
        cos_theta = torch.max(cos_theta, dim=0)[0]  # b*f
        theta = torch.acos(torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps))

        if target is None:
            return cos_theta

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)

        selected = torch.where(theta > self.threshold, torch.zeros_like(one_hot), one_hot)

        logits = torch.cos(torch.where(selected.bool(), theta + self.m, theta))
        logits *= self.s

        return logits, cos_theta


class CurricularFace(nn.Module):
    def __init__(self, in_features, out_features, m = 0.5, s = 64.):
        super(CurricularFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = nn.Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)

    @staticmethod
    def l2_norm(input, axis=1):
        norm = torch.norm(input, 2, axis, True)
        output = torch.div(input, norm)
        return output

    def forward(self, embbedings, label):
        embbedings = self.l2_norm(embbedings, axis = 1)
        kernel_norm = self.l2_norm(self.kernel, axis = 0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output, origin_cos * self.s


class AdaCos(nn.Module):
    def __init__(self, in_features, out_features, m=0.50, theta_zero=math.pi/4, *args, **kwargs):
        super(AdaCos, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.theta_zero = theta_zero
        self.s = math.log(out_features - 1) / math.cos(theta_zero)
        self.m = m
        self.kernel = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.kernel)

    def forward(self, input, label):
        # normalize features
        x = F.normalize(input)
        # normalize kernels
        W = F.normalize(self.kernel)
        # dot product
        logits = F.linear(x, W)
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(
                self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            theta_med = torch.median(theta)
            self.s = torch.log(
                B_avg) / torch.cos(torch.min(self.theta_zero * torch.ones_like(theta_med), theta_med))
        output *= self.s
        return output


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
                "query_id": format_video_id(c.query_id, Dataset.QUERIES) if do_format_video_id else c.query_id,
                "ref_id": format_video_id(c.ref_id, Dataset.REFS) if do_format_video_id else c.ref_id,
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
        cls, candidates: Collection["CustomCandidatePair"], file: Union[str, TextIO], do_format_video_id: bool = True,
    ):
        df = cls.to_dataframe(candidates, do_format_video_id)
        df.to_csv(file, index=False)

    @classmethod
    def read_csv(cls, file: Union[str, TextIO], do_format_video_id: bool = True) -> List["CustomCandidatePair"]:
        df = pd.read_csv(file)
        pairs = []
        for _, row in df.iterrows():
            query_id = format_video_id(row.query_id, Dataset.QUERIES) if do_format_video_id else row.query_id
            ref_id = format_video_id(row.ref_id, Dataset.REFS) if do_format_video_id else row.ref_id
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
        return match.matches[argmax].score, match.matches[argmax].query_timestamps, match.matches[argmax].ref_timestamps

    def score(self, match: PairMatches) -> CustomCandidatePair:
        score, query_max_idx, ref_max_idx = self.aggregate(match)
        return CustomCandidatePair(
            query_id=match.query_id, ref_id=match.ref_id, score=score,
            query_max_idx=query_max_idx[0], ref_max_idx=ref_max_idx[0],
        )


class CustomCandidateGeneration:
    def __init__(self, references: list[VideoFeature], aggregation: ScoreAggregation):
        self.aggregation = aggregation
        dim = references[0].dimensions()
        self.index = CustomVideoIndex(dim)
        self.index.add(references)

    def query(self, queries: list[VideoFeature], global_k: int) -> list[CustomCandidatePair]:
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
    """hstack/vstack, 回転（90, 180, 270度）, 左右反転, 上下反転の加工用TTA"""
    def __init__(self, base_transforms=None):
        super().__init__()
        self.base_transforms = base_transforms

    def forward(self, x) -> torch.Tensor:
        *_, h, w = x.shape

        x_top = x[..., :h // 2, :]  # top
        x_bottom = x[..., h // 2:, :]  # bottom
        x_left = x[..., :, :w // 2]  # left
        x_right = x[..., :, w // 2:]  # right

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
    """hstack + vstack の加工用TTA"""
    def __init__(self, base_transforms=None):
        super().__init__()
        self.base_transforms = base_transforms

    def forward(self, x) -> torch.Tensor:
        *_, h, w = x.shape

        x_top_left = x[..., :h // 2, :w // 2]  # top_left
        x_top_right = x[..., :h // 2, w // 2:]  # top_right
        x_bottom_left = x[..., h // 2:, :w // 2]  # bottom_left
        x_bottom_right = x[..., h // 2:, w // 2:]  # bottom_right

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


@dataclass
class Stream:
    profile: str
    width: int
    height: int
    coded_width: int
    coded_height: int
    pix_fmt: str
    level: int
    chroma_location: Optional[str]
    r_frame_rate: str
    avg_frame_rate: str
    time_base: str
    duration_ts: int
    duration: float
    bit_rate: int
    nb_frames: int
    sample_aspect_ratio: Optional[str]  # None: (probably) not augmented
    display_aspect_ratio: Optional[str]


@dataclass
class Format:
    filename: str
    nb_streams: int  # 1: video only, 2: video + audio
    duration: float
    size: int
    bit_rate: int


@dataclass
class FFProbeMetadata:
    stream: Stream
    format: Format


def get_video_metadata(path: str):
    import ffmpeg
    probe = ffmpeg.probe(path)
    for stream in probe["streams"]:
        if stream["codec_type"] == "video":
            break
    format = probe["format"]

    return FFProbeMetadata(
        stream=Stream(
            profile=stream["profile"],
            width=int(stream["width"]),
            height=int(stream["height"]),
            coded_width=int(stream["coded_width"]),
            coded_height=int(stream["coded_height"]),
            pix_fmt=str(stream["pix_fmt"]),
            level=int(stream["level"]),
            chroma_location=stream.get("chroma_location"),
            r_frame_rate=str(stream["r_frame_rate"]),
            avg_frame_rate=str(stream["avg_frame_rate"]),
            time_base=str(stream["time_base"]),
            duration_ts=int(stream["duration_ts"]),
            duration=float(stream["duration"]),
            bit_rate=int(stream["bit_rate"]),
            nb_frames=int(stream["nb_frames"]),
            sample_aspect_ratio=stream.get("sample_aspect_ratio"),
            display_aspect_ratio=stream.get("display_aspect_ratio"),
        ),
        format=Format(
            filename=str(format["filename"]),
            nb_streams=int(format["nb_streams"]),
            duration=float(format["duration"]),
            size=int(format["size"]),
            bit_rate=int(format["bit_rate"]),
        )
    )
