# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import itertools
import logging
import os
from pathlib import Path
from typing import Iterable, List, Tuple
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
from src.vsc.localization import (
    VCSLLocalizationCandidateScore,
    VCSLLocalizationMaxSim,
    VCSLLocalizationMatchScore,
)

import torch.nn as nn
from src.model import create_model_in_runtime, create_model_in_runtime_2
from src.metadata import FFProbeMetadata, get_video_metadata
from src.tta import TTA24ViewsTransform, TTA30ViewsTransform, TTA4ViewsTransform, TTA5ViewsTransform
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


def localize_and_verify(
    queries: List[VideoFeature],
    refs: List[VideoFeature],
    candidates: List[CandidatePair],
    localize_per_query: float = 5.0,
    model_type: str = "TN",
    tn_max_step=5,
    min_length=4,
    concurrency=16,
    similarity_bias=0.5,
    tn_top_k=5,
    max_path=10,
    min_sim=0.2,
    max_iou=0.3,
    discontinue=3,
    sum_sim=8,
    ave_sim=0.3,
    diagonal_thres=10,
    iou_thresh=0.9,
    min_bins=1,
    max_peaks=100,
    min_peaks=10,
    # discontinue=3, min_sim=0.2, min_length=5, max_iou=0.3  # DTW
    # discontinue=3, min_sim=0.0, min_length=5, max_iou=0.3, sum_sim=8, ave_sim=0.3, diagonal_thres=10,  # DP
    # iou_thresh=0.9, min_sim=0.0, min_bins=1, max_peaks=100, min_peaks=10,  # HV
) -> List[Match]:
    num_to_localize = int(len(queries) * localize_per_query)
    candidates = candidates[:num_to_localize]

    alignment = VCSLLocalizationMatchScore(
        queries,
        refs,
        model_type=model_type,
        tn_max_step=tn_max_step,
        min_length=min_length,
        concurrency=concurrency,
        similarity_bias=similarity_bias,
        tn_top_k=tn_top_k,
        max_path=max_path,
        min_sim=min_sim,
        max_iou=max_iou,
        discontinue=discontinue,
        sum_sim=sum_sim,
        ave_sim=ave_sim,
        diagonal_thres=diagonal_thres,
        iou_thresh=iou_thresh,
        min_bins=min_bins,
        max_peaks=max_peaks,
        min_peaks=min_peaks,
    )

    matches = []
    logger.info("Aligning %s candidate pairs", len(candidates))
    BATCH_SIZE = 512
    i = 0
    while i < len(candidates):
        batch = candidates[i: i + BATCH_SIZE]
        matches.extend(alignment.localize_all(batch))
        i += len(batch)
        logger.info(
            "Aligned %d pairs of %d; %d predictions so far",
            i,
            len(candidates),
            len(matches),
        )
    return matches


def match(
    queries: List[VideoFeature],
    refs: List[VideoFeature],
    output_file: str = None,
    return_results: bool = False,
    model_type: str = "TN",
    tn_max_step=5,
    min_length=4,
    concurrency=16,
    similarity_bias=0.5,
    tn_top_k=5,
    max_path=10,
    min_sim=0.2,
    max_iou=0.3,
    discontinue=3,
    sum_sim=8,
    ave_sim=0.3,
    diagonal_thres=10,
    iou_thresh=0.9,
    min_bins=1,
    max_peaks=100,
    min_peaks=10,
    retrieve_per_query: float = 1200.0,
    candidates_per_query: float = 25.0,
) -> Tuple[str, str]:
    # Search
    candidates = search(queries, refs, retrieve_per_query=retrieve_per_query, candidates_per_query=candidates_per_query)
    # os.makedirs(output_path, exist_ok=True)
    # candidate_file = os.path.join(output_path, "candidates.csv")
    # CandidatePair.write_csv(candidates, candidate_file)
    # candidates = CandidatePair.read_csv(candidate_file)

    # Localize and verify
    matches = localize_and_verify(
        queries,
        refs,
        candidates,
        model_type=model_type,
        tn_max_step=tn_max_step,
        min_length=min_length,
        concurrency=concurrency,
        similarity_bias=similarity_bias,
        tn_top_k=tn_top_k,
        max_path=max_path,
        min_sim=min_sim,
        max_iou=max_iou,
        discontinue=discontinue,
        sum_sim=sum_sim,
        ave_sim=ave_sim,
        diagonal_thres=diagonal_thres,
        iou_thresh=iou_thresh,
        min_bins=min_bins,
        max_peaks=max_peaks,
        min_peaks=min_peaks,
    )
    # matches_file = os.path.join(output_path, "matches.csv")
    Match.write_csv(matches, output_file, drop_dup=True)

    if return_results:
        return candidates, matches


def worker_process(args, rank, world_size, output_filename, model_name, ref_path, noise_path, pca_matrix_path):
    logger.info(f"Starting worker {rank} of {world_size}.")
    device = get_device(args, rank, world_size)

    logger.info("Loading model")
    if model_name == "isc":
        model, transforms = create_model_in_runtime(transforms_device=device)
    elif model_name == "vit":
        model, transforms = create_model_in_runtime_2(transforms_device=device)
    else:
        raise ValueError(f"Model {args.model} is not supported")

    model = model.to(device).eval()

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
    for vf in run_inference(loader, model, device, transforms):
        queries.append(vf)
        progress.update()

    del loader
    del model
    del dataset

    if noise_path:
        stride = args.stride if args.stride is not None else int(args.fps)
        pca_matrix = faiss.read_VectorTransform(pca_matrix_path)
        queries = sliding_pca(queries=queries, mat=pca_matrix, stride=stride)

        queries = score_normalize(
            queries,
            load_features(noise_path),
            beta=1.2,
        )
    
    os.makedirs(output_filename.replace('subset_matches.csv', 'output'), exist_ok=True)
    # store_features(output_filename.replace('subset_matches.csv', f'output/subset_queries_{model_name}.npz'), queries)
    refs = load_features(ref_path)

    match(
        queries=queries,
        refs=refs,
        output_file=output_filename.replace('subset_matches.csv', f'output/matches_{model_name}.csv'),
        tn_max_step=5,
        min_length=3,
        tn_top_k=2,
        max_path=200,
        min_sim=0.1,
        max_iou=1.0,
    )


@torch.no_grad()
def run_inference(dataloader, model, device, transforms=None, tta=True) -> Iterable[VideoFeature]:
    name = None
    embeddings = []
    timestamps = []

    for batch in dataloader:
        names = batch["name"]
        assert names[0] == names[-1]  # single-video batches
        if name is not None and name != names[0]:
            timestamps = np.concatenate(timestamps, axis=0)
            feature = np.concatenate(embeddings, axis=0)
            if tta:
                timestamps = timestamps.reshape(-1, num_views).transpose(1, 0).ravel()
                feature = feature.reshape(-1, num_views, feature.shape[1]).transpose(1, 0, 2).reshape(-1, feature.shape[1])
            yield VideoFeature(
                video_id=name,
                timestamps=timestamps,
                feature=feature,
            )
            embeddings = []
            timestamps = []
        name = names[0]
        if transforms is not None:
            if tta:
                metadatas = dataloader.dataset.metadatas
                # if metadatas[name].format.nb_streams == 2:
                if metadatas[name].stream.time_base == '1/10240':
                    tta_transforms = TTA4ViewsTransform(transforms)
                else:
                    tta_transforms = TTA5ViewsTransform(transforms)
                img = tta_transforms(batch["input"].to(device))
            else:
                img = transforms(batch["input"].to(device))
            num_views = img.shape[0] // batch["input"].shape[0]
            ts = batch["timestamp"].numpy()
            ts = np.repeat(ts, num_views)
        else:
            img = batch["input"].to(device)
            ts = batch["timestamp"].numpy()
            num_views = 1

        emb = model(img).cpu().numpy()
        embeddings.append(emb)
        timestamps.append(ts)

    timestamps = np.concatenate(timestamps, axis=0)
    feature = np.concatenate(embeddings, axis=0)
    if tta:
        timestamps = timestamps.reshape(-1, num_views).transpose(1, 0).ravel()
        feature = feature.reshape(-1, num_views, feature.shape[1]).transpose(1, 0, 2).reshape(-1, feature.shape[1])
    yield VideoFeature(
        video_id=name,
        timestamps=timestamps,
        feature=feature,
    )


@torch.no_grad()
def run_inference_ensemble(dataloader, model_list, device, transforms_list=None, weight_list=None, tta=True) -> Iterable[VideoFeature]:
    name = None
    embeddings = []
    timestamps = []

    if weight_list is None:
        weight_list = [1.0] * len(model_list)

    for batch in dataloader:
        names = batch["name"]
        assert names[0] == names[-1]  # single-video batches
        if name is not None and name != names[0]:
            timestamps = np.concatenate(timestamps, axis=0)
            feature = np.concatenate(embeddings, axis=0)
            if tta:
                timestamps = timestamps.reshape(-1, num_views).transpose(1, 0).ravel()
                feature = feature.reshape(-1, num_views, feature.shape[1]).transpose(1, 0, 2).reshape(-1, feature.shape[1])
            yield VideoFeature(
                video_id=name,
                timestamps=timestamps,
                feature=feature,
            )
            embeddings = []
            timestamps = []
        name = names[0]

        imgs = []
        for transforms in transforms_list:
            if tta:
                metadatas = dataloader.dataset.metadatas
                # if metadatas[name].format.nb_streams == 2:
                if metadatas[name].stream.time_base == '1/10240':
                    tta_transforms = TTA4ViewsTransform(transforms)
                else:
                    tta_transforms = TTA5ViewsTransform(transforms)
                img = tta_transforms(batch["input"].to(device))
            else:
                img = transforms(batch["input"].to(device))
            imgs.append(img)

        emb_cat = []
        for img, model, weight in zip(imgs, model_list, weight_list):
            emb = model(img).cpu().numpy() * weight
            emb_cat.append(emb)
        emb = np.concatenate(emb_cat, axis=1)
        embeddings.append(emb)

        num_views = imgs[0].shape[0] // batch["input"].shape[0]
        ts = batch["timestamp"].numpy()
        ts = np.repeat(ts, num_views)
        timestamps.append(ts)

    timestamps = np.concatenate(timestamps, axis=0)
    feature = np.concatenate(embeddings, axis=0)
    if tta:
        timestamps = timestamps.reshape(-1, num_views).transpose(1, 0).ravel()
        feature = feature.reshape(-1, num_views, feature.shape[1]).transpose(1, 0, 2).reshape(-1, feature.shape[1])

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
