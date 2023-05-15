#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Inference script.

This is split into inference and inference_impl to avoid initializing cuda
before processes are created that may use cuda, which can lead to errors
in some runtime environments.

We import inference_impl, which imports libraries that may initialize cuda,
in two circumstances: from worker processes after the main process has
forked workers, or from the main process after worker processes have been
joined.
"""

import argparse
import logging
import os
import tempfile
from pathlib import Path
from typing import List

import pandas as pd
import tqdm
from src.inference import Accelerator, VideoReaderType
from src.vsc.storage import load_features, store_features
from torch import multiprocessing

parser = argparse.ArgumentParser()
inference_parser = parser.add_argument_group("Inference")
inference_parser.add_argument("--batch_size", type=int, default=32)
inference_parser.add_argument("--distributed_rank", type=int, default=0)
inference_parser.add_argument("--distributed_size", type=int, default=1)
inference_parser.add_argument("--processes", type=int, default=1)
inference_parser.add_argument(
    "--accelerator", choices=[x.name.lower() for x in Accelerator], default="cpu"
)
inference_parser.add_argument("--output_path", nargs="+")
inference_parser.add_argument("--scratch_path", required=False)

dataset_parser = parser.add_argument_group("Dataset")
# multiple dataset path
dataset_parser.add_argument("--dataset_paths", nargs="+")
dataset_parser.add_argument("--gt_path")
dataset_parser.add_argument("--fps", default=1, type=float)
dataset_parser.add_argument("--len_cap", type=int)
dataset_parser.add_argument("--read_type", default="tensor", type=str)
dataset_parser.add_argument("--video_extensions", default="mp4")
dataset_parser.add_argument(
    "--video_reader", choices=[x.name for x in VideoReaderType], default="FFMPEGPY"
)
dataset_parser.add_argument("--ffmpeg_path", default="ffmpeg")
dataset_parser.add_argument("--tta", action="store_true")
dataset_parser.add_argument(
    "--mode", default="whole", choices=["whole", "eval", "test", "tune"]
)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("inference.py")
logger.setLevel(logging.INFO)


def main(args):
    success = False
    if args.processes > 1 and args.distributed_size > 1:
        raise Exception(
            "Set either --processes (single-machine distributed) or "
            "both --distributed_size and --distributed_rank (arbitrary "
            "distributed)"
        )

    if args.video_reader == "DECORD":
        import subprocess

        subprocess.run(
            ["pip", "install", "wheels/decord-0.6.0-py3-none-manylinux2010_x86_64.whl"],
            check=True,
        )

    with tempfile.TemporaryDirectory() as tmp_path:
        splits = [
            "_".join(p.split("/")[-2:]) for p in args.dataset_paths
        ]  # ./vsc/eval_subset/reference -> eval_subset_reference
        os.makedirs(args.output_path, exist_ok=True)
        if args.scratch_path:
            os.makedirs(args.scratch_path, exist_ok=True)
        else:
            args.scratch_path = tmp_path
        if args.processes > 1:
            processes = []
            logger.info(f"Spawning {args.processes} processes")
            accelerator = Accelerator[args.accelerator.upper()]
            backend = "nccl" if accelerator == Accelerator.CUDA else "gloo"
            multiprocessing.set_start_method("spawn")
            worker_files = []
            try:
                for rank in range(args.processes):
                    output_files = [
                        os.path.join(args.scratch_path, f"{split}_{rank}.npz")
                        for split in splits
                    ]
                    worker_files.append(output_files)
                    p = multiprocessing.Process(
                        target=distributed_worker_process,
                        args=(
                            args,
                            rank,
                            args.processes,
                            backend,
                            output_files,
                            args.dataset_paths,
                            args.tta,
                        ),
                    )
                    processes.append(p)
                    p.start()
                worker_success = []
                for p in processes:
                    p.join()
                    worker_success.append(p.exitcode == os.EX_OK)
                success = all(worker_success)
            finally:
                for p in processes:
                    p.kill()
            if success:

                def merge_feature_files(filenames, output_filename: str) -> int:
                    features = []
                    for fn in filenames:
                        features.extend(load_features(fn))
                    features = sorted(features, key=lambda x: x.video_id)
                    store_features(output_filename, features)
                    return len(features)

                output_files_each_split = [list(x) for x in zip(*worker_files)]
                for files, split in zip(output_files_each_split, splits):
                    output_file = os.path.join(args.output_path, f"{split}.npz")
                    num_files = merge_feature_files(files, output_file)
                    logger.info(
                        f"Features for {num_files} videos saved to {output_file}"
                    )

        else:
            output_files = [
                os.path.join(args.output_path, f"{split}.npz") for split in splits
            ]
            worker_process(
                args,
                args.distributed_rank,
                args.distributed_size,
                output_files,
                args.dataset_paths,
                args.tta,
            )
            success = True

    if success:
        logger.info("Inference succeeded.")
    else:
        logger.error("Inference FAILED!")

    if not success or args.gt_path is None:
        return

    logger.info("Evaluating results")

    evaluate(
        queries=load_features(os.path.join(args.output_path, f"{splits[0]}.npz")),
        refs=load_features(os.path.join(args.output_path, f"{splits[1]}.npz")),
        noises=load_features(os.path.join(args.output_path, f"{splits[-1]}.npz")),
        gt_path=args.gt_path,
        output_path=args.output_path,
    )


def distributed_worker_process(pargs, rank, world_size, backend, *args, **kwargs):
    from torch import distributed

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "19529"
    distributed.init_process_group(backend, rank=rank, world_size=world_size)
    worker_process(pargs, rank, world_size, *args, **kwargs)


def worker_process(
    args,
    rank,
    world_size,
    output_files: List[str],
    dataset_paths: List[str],
    tta: bool = False,
):
    from src.inference_impl import VideoDataset, get_device, run_inference
    from src.model import create_model_in_runtime
    from torch.utils.data import DataLoader

    logger.info(f"Starting worker {rank} of {world_size}.")
    device = get_device(args, rank, world_size)
    logger.info("Loading model")
    model, transforms = create_model_in_runtime(transforms_device=device)
    model = model.to(device).eval()
    logger.info("Setting up dataset")
    extensions = args.video_extensions.split(",")
    video_reader = VideoReaderType[args.video_reader.upper()]

    if tta:
        if len(dataset_paths) == 3:
            do_tta_list = [
                True,
                False,
                False,
            ]
        elif len(dataset_paths) == 4:
            do_tta_list = [
                True,
                False,
                True,
                False,
            ]
        elif len(dataset_paths) == 1:
            do_tta_list = [
                True,
            ]
        else:
            raise ValueError("TTA requires 3 or 4 datasets")
    else:
        do_tta_list = [False] * len(dataset_paths)

    for output_filename, dataset_path, do_tta in zip(
        output_files, dataset_paths, do_tta_list
    ):
        batch_size = 1 if do_tta else args.batch_size
        dataset = VideoDataset(
            dataset_path,
            fps=args.fps,
            read_type=args.read_type,
            batch_size=batch_size,
            extensions=extensions,
            distributed_world_size=world_size,
            distributed_rank=rank,
            video_reader=video_reader,
            ffmpeg_path=args.ffmpeg_path,
            filter_by_asr=do_tta,
        )
        loader = DataLoader(dataset, batch_size=None, pin_memory=device.type == "cuda")

        progress = tqdm.tqdm(total=dataset.num_videos())
        video_features = []
        for vf in run_inference(loader, model, device, transforms, do_tta):
            video_features.append(vf)
            progress.update()

        store_features(output_filename, video_features)


def evaluate(queries, refs, noises, gt_path, output_path, sn_method="SN"):
    import faiss
    from src.inference_impl import match
    from src.postproc import sliding_pca, sliding_pca_with_ref
    from src.score_normalization import (
        negative_embedding_subtraction,
        score_normalize_with_ref,
    )
    from src.vsc.metrics import (
        CandidatePair,
        Match,
        average_precision,
        evaluate_matching_track,
    )

    stride = 2
    video_features, pca_matrix = sliding_pca_with_ref(
        queries=queries,
        refs=refs,
        noises=noises,
        stride=stride,
    )
    queries = video_features["query"]
    refs = video_features["ref"]
    noises = video_features["noise"]
    store_features(Path(output_path) / "noise.npz", noises)
    faiss.write_VectorTransform(pca_matrix, str(Path(output_path) / "pca_matrix.bin"))

    if sn_method == "SN":
        queries, refs = score_normalize_with_ref(
            queries=queries,
            refs=refs,
            score_norm_refs=noises,
            beta=1.2,
        )

    else:
        queries, refs = negative_embedding_subtraction(
            queries=queries,
            refs=refs,
            score_norm_refs=noises,
            pre_l2_normalize=False,
            post_l2_normalize=False,
            beta=0.8,
            k=10,
            alpha=2.0,
        )

    store_features(Path(output_path) / "processed_ref.npz", refs)

    candidates, matches = match(
        queries=queries,
        refs=refs,
        output_file=None,
        return_results=True,
        similarity_bias=0.5 if sn_method == "SN" else 0.0,
        model_type="TN",
        tn_max_step=5,
        min_length=3,
        tn_top_k=2,
        max_path=200,
        min_sim=0.1,
        max_iou=1.0,
    )

    gt_matches = Match.read_csv(gt_path, is_gt=True)
    ap = average_precision(CandidatePair.from_matches(gt_matches), candidates)

    candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
    CandidatePair.write_csv(candidates, Path(output_path) / "candidates.csv")

    match_file = Path(output_path) / "matches.csv"
    Match.write_csv(matches, match_file, drop_dup=True)
    match_metrics = evaluate_matching_track(gt_path, match_file)

    logger.info(f"uAP: {ap.ap:.4f}, segmentAP: {match_metrics.segment_ap.ap:.4f}")


def tune(
    queries,
    refs,
    noises,
    gt_path,
    output_path,
):
    from sklearn.model_selection import ParameterGrid
    from src.inference_impl import match
    from src.score_normalization import (
        negative_embedding_subtraction,
        score_normalize_with_ref,
    )
    from src.vsc.metrics import (
        CandidatePair,
        Match,
        average_precision,
        evaluate_matching_track,
    )

    param_grid = [
        {
            "sn_method": ["SN"],
            "beta": [1.2],
            # "k": [10],
            # "alpha": [2.0],
            "similarity_bias": [0.5],
            "retrieve_per_query": [1200],
            "candidates_per_query": [25],
            "tn_max_step": [5],
            "min_length": [3],
            "tn_top_k": [2],
            "max_path": [70, 100, 150, 200],
            "min_sim": [0.1],
            "max_iou": [1.0],
        }
    ]

    rows = []

    for params in ParameterGrid(param_grid):

        if params["sn_method"] == "SN":
            norm_queries, norm_refs = score_normalize_with_ref(
                queries=queries,
                refs=refs,
                score_norm_refs=noises,
                beta=params["beta"],
            )

        else:
            norm_queries, norm_refs = negative_embedding_subtraction(
                queries=queries,
                refs=refs,
                score_norm_refs=noises,
                pre_l2_normalize=False,
                post_l2_normalize=False,
                beta=params["beta"],
                k=params["k"],
                alpha=params["alpha"],
            )

        candidates, matches = match(
            queries=norm_queries,
            refs=norm_refs,
            output_file=None,
            return_results=True,
            similarity_bias=params["similarity_bias"],
            model_type="TN",
            tn_max_step=params["tn_max_step"],
            min_length=params["min_length"],
            concurrency=16,
            tn_top_k=params["tn_top_k"],
            max_path=params["max_path"],
            min_sim=params["min_sim"],
            max_iou=params["max_iou"],
            discontinue=3,
            sum_sim=8,
            ave_sim=0.3,
            diagonal_thres=10,
            iou_thresh=0.9,
            min_bins=1,
            max_peaks=100,
            min_peaks=10,
            retrieve_per_query=params["retrieve_per_query"],
            candidates_per_query=params["candidates_per_query"],
        )

        gt_matches = Match.read_csv(gt_path, is_gt=True)
        ap = average_precision(CandidatePair.from_matches(gt_matches), candidates)

        # candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
        # CandidatePair.write_csv(candidates, Path(output_path) / 'candidates.csv')

        print(f"{len(matches)=}")
        matches = pd.DataFrame(matches)
        matches = matches[
            [
                "query_id",
                "ref_id",
                "query_start",
                "query_end",
                "ref_start",
                "ref_end",
                "score",
            ]
        ]
        matches = matches.sort_values("score", ascending=False)

        match_file = Path(output_path) / "tune_matches.csv"
        matches.to_csv(match_file, index=False)
        match_metrics = evaluate_matching_track(gt_path, match_file)

        rows.append(
            {
                "uAP": ap.ap,
                "segmentAP": match_metrics.segment_ap.ap,
                "len_matches": len(matches),
                **params,
            }
        )
        print(rows[-1])

    df = pd.DataFrame(rows)
    print(df.to_csv())
    df.to_csv("tuning_result.csv", index=False)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.mode == "eval":
        evaluate(
            queries=load_features(os.path.join(args.output_path, f"train_query.npz")),
            refs=load_features(os.path.join(args.output_path, f"train_reference.npz")),
            noises=load_features(os.path.join(args.output_path, f"test_reference.npz")),
            gt_path=args.gt_path,
            output_path=args.output_path,
            sn_method="SN",
        )
    elif args.mode == "test":
        evaluate(
            queries=load_features(
                os.path.join(args.output_path, f"phase_2_uB82_query.npz")
            ),
            refs=load_features(os.path.join(args.output_path, f"test_reference.npz")),
            noises=load_features(
                os.path.join(args.output_path, f"train_reference.npz")
            ),
            gt_path=args.gt_path,
            output_path=args.output_path,
            sn_method="SN",
        )
    elif args.mode == "tune":
        tune(
            queries=load_features(os.path.join(args.output_path, f"train_query.npz")),
            refs=load_features(os.path.join(args.output_path, f"train_reference.npz")),
            noises=load_features(os.path.join(args.output_path, f"test_reference.npz")),
            gt_path=args.gt_path,
            output_path=args.output_path,
        )
    else:
        main(args)
