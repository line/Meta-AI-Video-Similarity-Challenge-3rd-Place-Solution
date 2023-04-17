#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace
from pathlib import Path
from shutil import copyfile, move
import os

import numpy as np
import pandas as pd

parser = ArgumentParser()
parser.add_argument(
    "--subset_proportion",
    help="Proportion of data to include in subset",
    type=float,
    default=0.01,
)
parser.add_argument(
    "--dataset",
    help="Dataset type",
    type=str,
    required=True,
    choices=["train", "test", "phase2"],
)


def main(args: Namespace):
    # Ensure that data has been downloaded to the correct place
    competition_data_dir = Path("/share-cvlab/yokoo/vsc/competition_data")
    # runtime_data_dir = Path(__file__).parent.parent / "data" / args.dataset
    runtime_data_dir =  Path("data") / args.dataset
    query_subset_dir = runtime_data_dir / "query"
    dataset_folder = competition_data_dir / args.dataset
    query_video_dir = dataset_folder / "query"

    query_metadata = dataset_folder / f"{args.dataset}_query_metadata.csv"
    reference_metadata = dataset_folder / f"{args.dataset}_reference_metadata.csv"

    for file in [
        competition_data_dir,
        dataset_folder,
        query_metadata,
        reference_metadata,
    ]:
        if not file.exists():
            raise FileExistsError(
                f"Error: {file} not found. Have you downloaded or symlinked the competition dataset into {competition_data_dir}?"
            )

    p_subset = args.subset_proportion

    # Get a list of actually downloaded videos in system
    video_ids = os.listdir(query_video_dir)
    video_ids = [elem.replace('.mp4', '') for elem in video_ids]
    
    query_metadata_df = pd.read_csv(query_metadata)
    # Filter out the listed video_ids that are not in the system
    query_metadata_df = query_metadata_df[query_metadata_df['video_id'].isin(video_ids)]
    # Choose a random subset of query IDs to include
    rng = np.random.RandomState(42)
    subset_query_ids = query_metadata_df.video_id.sample(
        int(np.ceil(query_metadata_df.shape[0] * p_subset)), random_state=rng
    )

    # Copy metadata files and chosen videoss

    subset_query_ids.sort_values().to_csv(
        runtime_data_dir / "query_subset.csv", index=False
    )
    for query_id in subset_query_ids:
        copyfile(
            query_video_dir / f"{query_id}.mp4",
            query_subset_dir / f"{query_id}.mp4",
        )
    copyfile(query_metadata, runtime_data_dir / "query_metadata.csv")
    copyfile(reference_metadata, runtime_data_dir / "reference_metadata.csv")

    if args.dataset == "train":
        train_ground_truth = dataset_folder / "train_matching_ground_truth.csv"
        copyfile(train_ground_truth, runtime_data_dir / train_ground_truth.name)
        ground_truth_subset_df = pd.read_csv(train_ground_truth).set_index("query_id")
        ground_truth_subset_df = ground_truth_subset_df.loc[
            ground_truth_subset_df.index.isin(subset_query_ids)
        ]
        ground_truth_subset_df.to_csv(runtime_data_dir / "subset_ground_truth.csv")

        ref_video_dir = dataset_folder / "reference"
        ref_subset_dir = runtime_data_dir / "reference"
        ref_subset_dir.mkdir(exist_ok=True)
        subset_ref_ids = ground_truth_subset_df.ref_id.unique()
        print(f"Copying {len(subset_ref_ids)} reference videos")
        for ref_id in subset_ref_ids:
            copyfile(
                ref_video_dir / f"{ref_id}.mp4",
                ref_subset_dir / f"{ref_id}.mp4",
            )

        noise_video_dir = competition_data_dir / "test" / "reference"
        noise_subset_dir = runtime_data_dir / "noise"
        noise_subset_dir.mkdir(exist_ok=True)
        noise_size = 128
        noise_subset_ids = [f"R2{idx + 1:05d}" for idx in range(noise_size)]
        print(f"Copying {len(noise_subset_ids)} noise videos (ex: {noise_subset_ids[:3]})")
        for noise_id in noise_subset_ids:
            copyfile(
                noise_video_dir / f"{noise_id}.mp4",
                noise_subset_dir / f"{noise_id}.mp4",
            )



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
