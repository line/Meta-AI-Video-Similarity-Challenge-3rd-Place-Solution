from pathlib import Path

import numpy as np
import pandas as pd

QUICKSTART_DIRECTORY = Path(__file__).parent.parent / "submission_quickstart"
QUERY_OUTPUT_FILE = QUICKSTART_DIRECTORY / "query_descriptors.npz"
REFERENCE_OUTPUT_FILE = QUICKSTART_DIRECTORY / "reference_descriptors.npz"
DATA_DIRECTORY = QUICKSTART_DIRECTORY.parent / "data" / "test"
QUERY_METADATA_PATH = DATA_DIRECTORY / "query_metadata.csv"
REFERENCE_METADATA_PATH = DATA_DIRECTORY / "reference_metadata.csv"


def generate_random_descriptors(all_video_ids) -> np.ndarray:
    # Initialize a reproducible random number generator
    rng = np.random.RandomState(42)

    # Choose a descriptor dimensionality
    n_dim = 16

    # Initialize return values
    video_ids = []
    timestamps = []
    descriptors = []

    # Generate random descriptors for each video
    for video_id in all_video_ids:
        # TODO: limit number of descriptors by video length, either
        # from reading in video or checking metadata file
        n_descriptors = rng.randint(low=5, high=15)
        descriptors.append(rng.standard_normal(size=(n_descriptors, n_dim)))

        # Insert random timestamps
        start_timestamps = 30 * rng.random(size=(n_descriptors, 1))
        end_timestamps = start_timestamps + 30 * rng.random(size=(n_descriptors, 1))

        timestamps.append(np.hstack([start_timestamps, end_timestamps]))
        video_ids.append(np.full(n_descriptors, video_id))

    video_ids = np.concatenate(video_ids)
    descriptors = np.concatenate(descriptors).astype(np.float32)
    timestamps = np.concatenate(timestamps).astype(np.float32)

    return video_ids, descriptors, timestamps


def main():
    query_video_ids = pd.read_csv(QUERY_METADATA_PATH).video_id.values
    reference_video_ids = pd.read_csv(REFERENCE_METADATA_PATH).video_id.values

    ### Generation of query descriptors happens here ######
    query_video_ids, query_descriptors, query_timestamps = generate_random_descriptors(
        query_video_ids
    )

    (
        reference_video_ids,
        reference_descriptors,
        reference_timestamps,
    ) = generate_random_descriptors(reference_video_ids)
    ##################################

    np.savez(
        QUERY_OUTPUT_FILE,
        video_ids=query_video_ids,
        features=query_descriptors,
        timestamps=query_timestamps,
    )

    np.savez(
        REFERENCE_OUTPUT_FILE,
        video_ids=reference_video_ids,
        features=reference_descriptors,
        timestamps=reference_timestamps,
    )


if __name__ == "__main__":
    main()
