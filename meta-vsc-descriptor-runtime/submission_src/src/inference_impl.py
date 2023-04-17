# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import itertools
import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import faiss

import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from src.inference import Accelerator, VideoReaderType
from src.video_reader.ffmpeg_video_reader import FFMpegVideoReader
from src.video_reader.ffmpeg_py_video_reader import FFMpegPyVideoReader
from src.vsc.index import VideoFeature
from src.vsc.storage import load_features, store_features
from src.vsc.candidates import CandidateGeneration, MaxScoreAggregation
from src.score_normalization import score_normalize
from src.vsc.metrics import CandidatePair, Match, average_precision, evaluate_matching_track

import torch.nn as nn
from src.model import create_model_in_runtime, create_model_in_runtime_2, create_copy_type_pred_model
from src.metadata import FFProbeMetadata, get_video_metadata
from src.tta import TTA4ViewsTransform, TTA5ViewsTransform, TTAHorizontalStackTransform, TTAVerticalStackTransform
from src.postproc import sliding_pca


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("inference_impl.py")
logger.setLevel(logging.INFO)


class VideoDataset(IterableDataset):
    """Decodes video frames at a fixed FPS via ffmpeg."""

    def __init__(
        self,
        path: str,
        fps: float,
        read_type: str = "pil",
        batch_size=None,
        img_transform=None,
        extensions=("mp4",),
        distributed_rank=0,
        distributed_world_size=1,
        video_reader=VideoReaderType.FFMPEG,
        ffmpeg_path="ffmpeg",
        filter_by_asr=False,
    ):
        assert distributed_rank < distributed_world_size
        self.path = path
        self.fps = fps
        self.read_type = read_type
        self.batch_size = batch_size
        self.img_transform = img_transform
        self.video_reader = video_reader
        self.ffmpeg_path = ffmpeg_path
        if len(extensions) == 1:
            filenames = glob.glob(os.path.join(path, f"*.{extensions[0]}"))
        else:
            filenames = glob.glob(os.path.join(path, "*.*"))
            filenames = (fn for fn in filenames if fn.rsplit(".", 1)[-1] in extensions)

        filenames = [name for name in filenames if Path(name).stem not in ['R102796', 'R133364']]

        if filter_by_asr:
            self.metadatas = {Path(path).stem: get_video_metadata(path) for path in filenames}
            filtered_filenames = [
                path for path in filenames if self.metadatas[Path(path).stem].stream.sample_aspect_ratio is not None
            ]
        else:
            self.metadatas = None
            filtered_filenames = filenames
        self.videos = sorted(filtered_filenames)
        print(f'filtered: #{len(filenames)} -> #{len(self.videos)}')

        if not self.videos:
            raise Exception("No videos found!")
        assert distributed_rank < distributed_world_size
        self.rank = distributed_rank
        self.world_size = distributed_world_size
        self.selected_videos = [
            (i, video)
            for (i, video) in enumerate(self.videos)
            if (i % self.world_size) == self.rank
        ]

    def num_videos(self) -> int:
        return len(self.selected_videos)

    def __iter__(self):
        for i, video in self.selected_videos:
            if self.batch_size:
                frames = self.read_frames(i, video)
                while True:
                    batch = list(itertools.islice(frames, self.batch_size))
                    if not batch:
                        break
                    yield default_collate(batch)
            else:
                yield from self.read_frames(i, video)

    def read_frames(self, video_id, video):
        video_name = os.path.basename(video)
        name = os.path.basename(video_name).split(".")[0]
        if self.video_reader == VideoReaderType.FFMPEG:
            reader = FFMpegVideoReader(
                video_path=video, required_fps=self.fps,
                output_type=self.read_type, ffmpeg_path=self.ffmpeg_path,
            )
        elif self.video_reader == VideoReaderType.FFMPEGPY:
            reader = FFMpegPyVideoReader(video_path=video, required_fps=self.fps, output_type=self.read_type)
        elif self.video_reader == VideoReaderType.DECORD:
            from src.video_reader.decord_video_reader import DecordVideoReader
            reader = DecordVideoReader(video_path=video, required_fps=self.fps, output_type=self.read_type)
        else:
            raise ValueError(f"VideoReaderType: {self.video_reader} not supported")
        for start_timestamp, end_timestamp, frame in reader.frames():
            if self.img_transform:
                frame = self.img_transform(frame)
            record = {
                "name": name,
                "timestamp": (np.array(start_timestamp) + np.array(end_timestamp)) / 2,
                "input": frame,
            }
            yield record


def should_use_cuda(args) -> bool:
    accelerator = Accelerator[args.accelerator.upper()]
    return accelerator == Accelerator.CUDA


def get_device(args, rank, world_size):
    if should_use_cuda(args):
        assert torch.cuda.is_available()
        num_devices = torch.cuda.device_count()
        if args.processes > num_devices:
            raise Exception(
                f"Asked for {args.processes} processes and cuda, but only "
                f"{num_devices} devices found"
            )
        if args.processes > 1 or world_size <= num_devices:
            device_num = rank
        else:
            device_num = 0
        torch.cuda.set_device(device_num)
        return torch.device("cuda", device_num)
    return torch.device("cpu")


def search(
    queries: List[VideoFeature],
    refs: List[VideoFeature],
    retrieve_per_query: float = 1200.0,
    candidates_per_query: float = 25.0,
) -> List[CandidatePair]:
    aggregation = MaxScoreAggregation()
    logger.info("Searching")
    cg = CandidateGeneration(refs, aggregation)
    num_to_retrieve = int(retrieve_per_query * len(queries))
    candidates = cg.query(queries, global_k=num_to_retrieve)
    num_candidates = int(candidates_per_query * len(queries))
    candidates = candidates[:num_candidates]
    logger.info("Got %d candidates", len(candidates))
    return candidates


def worker_process(args, rank, world_size, output_filename):
    logger.info(f"Starting worker {rank} of {world_size}.")
    device = get_device(args, rank, world_size)

    logger.info("Loading model")
    model, transforms = create_model_in_runtime(transforms_device=device)
    model = model.to(device).eval().half()

    copytype_model = create_copy_type_pred_model(transforms_device=device)
    copytype_model = copytype_model.to(device).eval()

    logger.info("Setting up dataset")
    extensions = args.video_extensions.split(",")
    video_reader = VideoReaderType[args.video_reader.upper()]
    dataset = VideoDataset(
        args.dataset_path,
        fps=args.fps,
        read_type=args.read_type,
        # img_transform=transforms,
        # batch_size=args.batch_size,
        batch_size=1,
        extensions=extensions,
        distributed_world_size=world_size,
        distributed_rank=rank,
        video_reader=video_reader,
        ffmpeg_path=args.ffmpeg_path,
        filter_by_asr=True,
    )
    loader = DataLoader(dataset, batch_size=None, pin_memory=device.type == "cuda")

    progress = tqdm.tqdm(total=dataset.num_videos())
    queries = []
    for vf in run_inference(
        dataloader=loader,
        model=model,
        transforms=transforms,
        tta=True,
        copytype_model=copytype_model,
        batch_size=args.batch_size,
    ):
        queries.append(vf)
        progress.update()

    del loader
    del model
    del dataset

    if args.score_norm_features:
        stride = args.stride if args.stride is not None else args.fps
        print('stride', stride, args.stride, args.fps)
        pca_matrix = faiss.read_VectorTransform(args.pca_matrix)
        queries = sliding_pca(queries=queries, mat=pca_matrix, stride=stride)

        queries = score_normalize(
            queries,
            load_features(args.score_norm_features),
            beta=1.0,
        )

    store_features(output_filename, queries)
    logger.info(
        f"Wrote worker {rank} features for {len(queries)} videos to {output_filename}"
    )


def video_level_loader(frame_level_dataloader: DataLoader) -> Iterable[Tuple[str, torch.Tensor, torch.Tensor]]:
    name, imgs, timestamps = None, [], []

    for frames in frame_level_dataloader:
        _names, _imgs, _ts = frames["name"], frames["input"], frames["timestamp"]

        assert _names[0] == _names[-1]  # single-video batches

        _name = _names[0]
        if name is not None and name != _name:
            yield name, torch.concat(imgs), torch.concat(timestamps)
            name, imgs, timestamps = _name, [_imgs], [_ts]
        else:
            name = _name
            imgs.append(_imgs)
            timestamps.append(_ts)

    yield name, torch.concat(imgs), torch.concat(timestamps)


def batch_forward(model: nn.Module, inputs: torch.Tensor, batch_size: int, transforms=None) -> torch.Tensor:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    outputs = []
    for i in range(0, len(inputs), batch_size):
        x = inputs[i: i+batch_size]

        if transforms is not None:
            x = transforms(x)

        if dtype == torch.float16:
            x = x.half()

        x = model(x.to(device))

        if isinstance(x, dict):
            x = {k: v.cpu() for k, v in x.items()}
        else:
            x = x.cpu()
        outputs.append(x)

    if isinstance(outputs[0], dict):
        output_dict = {}
        for k in outputs[0]:
            output_dict[k] = torch.cat([d[k] for d in outputs], dim=0)
        return output_dict
    else:
        return torch.cat(outputs, dim=0)


@torch.no_grad()
def run_inference(dataloader, model, transforms=None, tta=True, copytype_model=None, batch_size=32) -> Iterable[VideoFeature]:
    for name, imgs, timestamps in video_level_loader(dataloader):
        n_imgs = len(imgs)

        _transforms = transforms

        # Extend transforms
        if _transforms is not None and tta:
            metadatas = dataloader.dataset.metadatas

            # Judge by metadata (hstack + vstack)
            if metadatas[name].stream.time_base == '1/10240':
                _transforms = TTA4ViewsTransform(_transforms)
            # Judge by copytype pred model
            elif copytype_model is not None:
                pred = batch_forward(copytype_model, imgs, batch_size=batch_size)
                assert len(pred["hstack"]) == len(imgs), f"{len(pred['hstack'])=} != {len(imgs)=}"
                assert len(pred["vstack"]) == len(imgs), f"{len(pred['vstack'])=} != {len(imgs)=}"

                hstack_prob = torch.median(pred["hstack"])
                vstack_prob = torch.median(pred["vstack"])

                thresh = 0.15

                if hstack_prob > thresh:
                    _transforms = TTAHorizontalStackTransform(_transforms)
                elif vstack_prob > thresh:
                    _transforms = TTAVerticalStackTransform(_transforms)
            # if copytype model is not given
            else:
                _transforms = TTA5ViewsTransform(_transforms)

        if _transforms is None:
            _transforms = torch.nn.Identity()

        feature = batch_forward(
            model,
            imgs,
            batch_size=batch_size,
            transforms=_transforms,
        ).numpy()

        num_views = len(feature) // n_imgs
        timestamps = np.tile(timestamps.numpy(), num_views)

        yield VideoFeature(
            video_id=name,
            timestamps=timestamps,
            feature=feature,
        )


def merge_feature_files(filenames: List[str], output_filename: str) -> int:
    features = []
    for fn in filenames:
        features.extend(load_features(fn))
    store_features(output_filename, features)
    return len(features)


def validate_total_descriptors(video_features: List[VideoFeature], meta: Optional[pd.DataFrame] = None):
    _ids = [_.video_id for _ in video_features]
    n_features = sum([_.feature.shape[0] for _ in video_features])

    if meta is None:
        meta_root = Path('/share-cvlab/yokoo/vsc/competition_data')
        meta_paths = [
            meta_root / 'train' / 'train_query_metadata.csv',
            meta_root / 'train' / 'train_reference_metadata.csv',
            meta_root / 'test' / 'test_query_metadata.csv',
            meta_root / 'test' / 'test_reference_metadata.csv',
        ]
        meta = pd.concat([pd.read_csv(path) for path in meta_paths])

    meta = meta.set_index("video_id").loc[_ids]
    total_seconds = meta.duration_sec.apply(np.ceil).sum()

    logger.info(
        f"Saw {n_features} vectors, max allowed is {total_seconds}"
    )
    if n_features > total_seconds:
        logger.info("*Warning*: Too many vectors")
