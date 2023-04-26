import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import decord
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: pd.DataFrame,
        n_labels: int,
        transform: Optional[Callable] = None,
        tensor_format: str = "CHW",
        random_relpos_offset: float = 0.0,
    ):
        self.dataset = dataset
        self.n_labels = n_labels
        self.onehot = torch.eye(n_labels, dtype=torch.float32)
        self.transform = transform
        self.tensor_format = tensor_format
        self.random_relpos_offset = random_relpos_offset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        row = self.dataset.iloc[i]

        num_threads = 0
        ctx = decord.cpu(0)

        reader = decord.VideoReader(
            row["video_path"],
            ctx=ctx,
            num_threads=num_threads,
        )
        n_frames = len(reader)

        relpos = row["relative_pos"]
        if self.random_relpos_offset > 0.0:
            relpos += (np.random.random() * 2 - 1) * self.random_relpos_offset

        pos = int((n_frames * relpos).round())

        try:
            frame = reader[pos].asnumpy()
        except:
            print(row.to_dict())
            raise

        tensor = torch.as_tensor(frame)

        if self.transform:
            tensor = tensor.permute(2, 0, 1)
            tensor = self.transform(tensor)
            tensor = tensor.permute(1, 2, 0)

        if self.tensor_format == "CHW":
            tensor = tensor.permute(2, 0, 1)

        label = self.onehot[row["label_idx"].tolist()].max(dim=0).values
        return tensor, label


class VscDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir_path: Path,
        labels: List[str],
        frames_per_video: int,
        batch_size: int,
        train_transform: Optional[Callable] = None,
        valid_transform: Optional[Callable] = None,
        tensor_format: str = "CHW",
        num_workers: int = 1,
        jit: bool = True,
    ):
        super().__init__()
        self.dataset_dir_path = dataset_dir_path
        self.labels = labels
        self.frames_per_video = frames_per_video
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.tensor_format = tensor_format
        self.num_workers = num_workers
        self.jit = jit

    def setup(self, stage: str):
        if stage == "fit":
            train = pd.read_parquet(self.dataset_dir_path / "train.parquet")
            valid = pd.read_parquet(self.dataset_dir_path / "valid.parquet")

            t_train = self.train_transform
            t_valid = self.valid_transform

            if self.jit:
                t_train = t_train and torch.jit.script(t_train)
                t_valid = t_valid and torch.jit.script(t_valid)

            # make dataset
            self.train = self._make_dataset(train, len(self.labels), t_train, 0.05)
            self.valid = self._make_dataset(valid, len(self.labels), t_valid, 0.0)

    def _gather_augly_metadata(self) -> pd.DataFrame:
        metadata = []
        for p in sorted(self.augly_metadata_path.glob("*")):
            with open(p) as f:
                m = json.load(f)
            video_id = p.stem.split("_", 1)[0]
            row = {
                "video_id": video_id,
                "video_path": (self.video_dir_path / f"{video_id}.mp4").as_posix(),
                "aug_type": m["name"],
            }
            metadata.append(row)
        return pd.DataFrame(metadata)

    def _make_dataset(
        self,
        metadata: List[dict],
        n_labels: int,
        transform: Optional[Callable],
        random_relpos_offset: float,
    ) -> pd.DataFrame:
        dataset = pd.DataFrame(metadata).join(
            pd.DataFrame(
                {
                    "relative_pos": np.linspace(
                        0.1, 0.9, self.frames_per_video, dtype=float
                    )
                }
            ),
            how="cross",
        )

        return Dataset(
            dataset, n_labels, transform, self.tensor_format, random_relpos_offset
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
