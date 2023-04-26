import argparse
import shutil
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms
from utils.configs import timm_model_config
from utils.dataset import VscDataModule
from utils.model import LitModel


def train(args: argparse.Namespace):
    cudnn.benchmark = True
    pl.seed_everything(args.seed, workers=True)

    labels = (
        pd.read_csv(args.dataset_dir_path / "label.csv")
        .sort_values("label_idx")
        .drop_duplicates()
        .sort_values("label_idx")["label"]
        .tolist()
    )

    mean, std, input_size = timm_model_config(args.model_name)

    train_transform = nn.Sequential(
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=mean, std=std),
        transforms.Resize(size=input_size),
    )

    valid_transform = nn.Sequential(
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=mean, std=std),
        transforms.Resize(size=input_size),
    )

    data_module = VscDataModule(
        dataset_dir_path=args.dataset_dir_path,
        labels=labels,
        frames_per_video=args.frames_per_video,
        train_transform=train_transform,
        valid_transform=valid_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        jit=False,
    )

    model = LitModel(
        model_name=args.model_name,
        labels=labels,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(__file__).parent / "training_log",
        monitor="valid_loss",
        save_last=True,
        save_top_k=1,
    )

    trainer = pl.Trainer(
        strategy=args.ddp_strategy,
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.epochs,
        num_sanity_val_steps=-1,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, datamodule=data_module)

    shutil.copy(
        checkpoint_callback.best_model_path,
        "model.ckpt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir-path", required=True, type=Path)
    parser.add_argument("--frames-per-video", required=True, type=int)

    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)

    parser.add_argument("--seed", default=sum(ord(x) for x in "vsc"))

    parser.add_argument("--devices", nargs="+", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--ddp-strategy", default="ddp")

    args = parser.parse_args()

    train(args)
