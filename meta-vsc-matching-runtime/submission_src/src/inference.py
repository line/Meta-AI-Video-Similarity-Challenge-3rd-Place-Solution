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
import enum
import glob
import logging
import os
import tempfile

from torch import multiprocessing


class InferenceTransforms(enum.Enum):
    # Aspect-ratio preserving resize to 288
    RESIZE_288 = enum.auto()
    # Resize the short edge to 320, then take the center crop
    RESIZE_320_CENTER = enum.auto()


class Accelerator(enum.Enum):
    CPU = enum.auto()
    CUDA = enum.auto()


class VideoReaderType(enum.Enum):
    FFMPEG = enum.auto()
    FFMPEGPY = enum.auto()
    DECORD = enum.auto()


parser = argparse.ArgumentParser()
inference_parser = parser.add_argument_group("Inference")
# inference_parser.add_argument("--torchscript_path", required=True)
inference_parser.add_argument("--batch_size", type=int, default=32)
inference_parser.add_argument("--distributed_rank", type=int, default=0)
inference_parser.add_argument("--distributed_size", type=int, default=1)
inference_parser.add_argument("--processes", type=int, default=1)
inference_parser.add_argument(
    "--transforms",
    choices=[x.name for x in InferenceTransforms],
    default="RESIZE_320_CENTER",
)
inference_parser.add_argument(
    "--accelerator", choices=[x.name.lower() for x in Accelerator], default="cpu"
)
inference_parser.add_argument("--output_file", required=True)
inference_parser.add_argument("--scratch_path", required=False)

dataset_parser = parser.add_argument_group("Dataset")
dataset_parser.add_argument("--dataset_path", required=True)
dataset_parser.add_argument("--fps", default=1, type=float)
dataset_parser.add_argument("--stride", type=int)
dataset_parser.add_argument("--read_type", default="pil", type=str)
dataset_parser.add_argument("--video_extensions", default="mp4")
dataset_parser.add_argument(
    "--video_reader", choices=[x.name for x in VideoReaderType], default="FFMPEG"
)
dataset_parser.add_argument("--ffmpeg_path", default="ffmpeg")
dataset_parser.add_argument("--score_norm_features", default=["./src/isc/test_noise.npz", "./src/vit/test_noise.npz"])
dataset_parser.add_argument("--reference_features", default=["./src/isc/test_processed_ref.npz", "./src/vit/test_processed_ref.npz"])
dataset_parser.add_argument("--model", default=["isc", "vit"])
dataset_parser.add_argument("--pca_matrix", default=["./src/isc/test_pca_matrix.bin", "./src/vit/test_pca_matrix.bin"])
dataset_parser.add_argument("--gt_path")


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
        subprocess.run(["pip", "install", "wheels/decord-0.6.0-py3-none-manylinux2010_x86_64.whl"], check=True)

    for model, ref_path, noise_path, pca_matrix in zip(args.model, args.reference_features, args.score_norm_features, args.pca_matrix):
        with tempfile.TemporaryDirectory() as tmp_path:
            os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
            if args.scratch_path:
                os.makedirs(args.scratch_path, exist_ok=True)
            else:
                args.scratch_path = tmp_path
            if args.processes > 1:
                processes = []
                logger.info(f"Spawning {args.processes} processes")
                accelerator = Accelerator[args.accelerator.upper()]
                backend = "nccl" if accelerator == Accelerator.CUDA else "gloo"
                # multiprocessing.set_start_method("spawn")
                try:
                    multiprocessing.set_start_method('spawn')
                except RuntimeError:
                    pass
                worker_files = []
                try:
                    for rank in range(args.processes):
                        worker_file = os.path.join(args.scratch_path, f"{rank}.npz")
                        worker_files.append(worker_file)
                        p = multiprocessing.Process(
                            target=distributed_worker_process,
                            args=(args, rank, args.processes, backend, worker_file, model, ref_path, noise_path, pca_matrix),
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
                    from .inference_impl import merge_feature_files  # @manual

                    num_files = merge_feature_files(worker_files, args.output_file)
                    logger.info(
                        f"Features for {num_files} videos saved to {args.output_file}"
                    )

            else:
                worker_process(
                    args, args.distributed_rank, args.distributed_size, args.output_file,
                    model, ref_path, noise_path, pca_matrix
                )
                success = True

    matches_list = glob.glob(f"{os.path.dirname(args.output_file)}/output/*.csv")
    from src.postproc import ensemble_match_results
    math_file = ensemble_match_results(matches_list, args.output_file)
    
    if success:
        logger.info("Inference succeeded.")
    else:
        logger.error("Inference FAILED!")


def distributed_worker_process(args, rank, world_size, backend, output_filename, model, ref_path, noise_path):
    from torch import distributed

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "19529"
    distributed.init_process_group(backend, rank=rank, world_size=world_size)
    worker_process(args, rank, world_size, output_filename, model, ref_path, noise_path, pca_matrix)


def worker_process(*args):
    # Late import: initialize cuda after worker spawn.
    from src.inference_impl import worker_process as worker_impl  # @manual

    return worker_impl(*args)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
