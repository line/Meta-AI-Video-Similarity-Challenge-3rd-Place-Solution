from pathlib import Path

import numpy as np
import pandas as pd

QUICKSTART_DIRECTORY = Path(__file__).parent.parent / "submission_quickstart"
DATA_DIRECTORY = QUICKSTART_DIRECTORY.parent / "data" / "test"
QUERY_METADATA_PATH = DATA_DIRECTORY / "query_metadata.csv"
REFERENCE_METADATA_PATH = DATA_DIRECTORY / "reference_metadata.csv"
MATCHES_OUTPUT_FILE = QUICKSTART_DIRECTORY / "full_matches.csv"


def generate_interval(row, rng):
    interval = np.sort(rng.random(2))
    start, end = row.duration_sec.values[0] * interval
    return start, end


def generate_random_matches(query_meta, ref_meta) -> pd.DataFrame:
    # Initialize a reproducible random number generator
    rng = np.random.RandomState(42)

    # Initialize return values
    matches = []

    # Generate random matches for each video
    for _ in range(query_meta.shape[0]):
        # Flip a coin!
        if rng.random() < 0.5:
            # Generate a random match
            query = query_meta.sample()
            query_start, query_end = generate_interval(query, rng)

            ref = ref_meta.sample()
            ref_start, ref_end = generate_interval(ref, rng)

            score = rng.random()

            matches.append(
                {
                    "query_id": query.video_id.values[0],
                    "ref_id": ref.video_id.values[0],
                    "query_start": query_start,
                    "query_end": query_end,
                    "ref_start": ref_start,
                    "ref_end": ref_end,
                    "score": score,
                }
            )

    return pd.DataFrame.from_records(matches)


def main():
    query_metadata = pd.read_csv(QUERY_METADATA_PATH)
    reference_metadata = pd.read_csv(REFERENCE_METADATA_PATH)

    # Generation of query matches happens here
    matches = generate_random_matches(query_metadata, reference_metadata)

    matches.to_csv(MATCHES_OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()
