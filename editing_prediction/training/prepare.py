import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

AUG_LABEL_MAPPING = {
    "rotate90": "rotate90_270",
    "rotate270": "rotate90_270",
    "rotate180": "rotate180_vflip",
    "vflip": "rotate180_vflip",
    "hflip": "other",
    "change_video_speed": "other",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir-path", required=True, type=Path)
    parser.add_argument("--augly-metadata-path", required=True, type=Path)
    args = parser.parse_args()

    seed = 1234
    dataset_dir = Path("./dataset")
    video_dir_path = args.video_dir_path
    augly_metadata_path = args.augly_metadata_path

    # Gather metadata
    metadata = []
    for p in sorted(augly_metadata_path.glob("*")):
        with open(p) as f:
            m = json.load(f)
        video_id = p.stem.split("_", 1)[0]
        row = {
            "video_id": video_id,
            "video_path": (video_dir_path / f"{video_id}.mp4").as_posix(),
            "aug_type": m.get("sub_name", m["name"]),
        }
        metadata.append(row)
    metadata = pd.DataFrame(metadata)

    # Assign label id
    aug_types = sorted(metadata["aug_type"].unique())
    aug_types = pd.DataFrame({"aug_type": aug_types})
    aug_types["label"] = aug_types["aug_type"].replace(AUG_LABEL_MAPPING)

    labels = sorted(aug_types["label"].unique())
    label_idx = pd.Series(np.arange(len(labels)), index=labels, name="label_idx")

    aug_label_mapping = aug_types.join(label_idx, on="label", how="left").set_index(
        "aug_type"
    )

    print("Labels")
    print(aug_label_mapping.to_markdown())

    metadata = metadata.join(aug_label_mapping, on="aug_type", how="left")

    # Grouping
    sorted_tuple = lambda x: tuple(sorted(x))
    metadata = (
        metadata.drop_duplicates()
        .groupby(["video_id", "video_path"])
        .agg(
            {
                "aug_type": sorted_tuple,
                "label": sorted_tuple,
                "label_idx": sorted_tuple,
            }
        )
        .reset_index()
    )

    # Random split
    label_counts = (
        metadata["label"]
        .value_counts()
        .reset_index(name="c")
        .rename(columns={"index": "label"})
    )
    label_counts["stratify_label"] = label_counts.apply(
        lambda x: x["label"] if x["c"] >= 10 else ("minor",), axis=1
    )
    stratifier = metadata.merge(label_counts, on="label")["stratify_label"]

    train, valid = train_test_split(
        metadata,
        test_size=0.2,
        stratify=stratifier,
        random_state=seed,
    )

    print(f"Train datset: {train.shape}")
    print(f"Valid datset: {valid.shape}")

    dataset_dir.mkdir(parents=True, exist_ok=True)

    aug_label_mapping.to_csv(dataset_dir / "label.csv", index=False)
    train.to_parquet(dataset_dir / "train.parquet", index=False)
    valid.to_parquet(dataset_dir / "valid.parquet", index=False)
