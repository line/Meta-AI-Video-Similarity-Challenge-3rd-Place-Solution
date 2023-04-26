import argparse
import csv
import glob
import json
import os
import random
import tempfile
from pathlib import Path

import decord
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from augly.video import (
    BlendVideos,
    Brightness,
    ColorJitter,
    Contrast,
    Crop,
    Grayscale,
    HFlip,
    HStack,
    InsertInBackground,
    Loop,
    Overlay,
    OverlayDots,
    OverlayEmoji,
    OverlayShapes,
    OverlayText,
    Pad,
    PerspectiveTransformAndShake,
    RandomAspectRatio,
    RandomBlur,
    RandomEncodingQuality,
    RandomNoise,
    RandomPixelization,
    RandomVideoSpeed,
    Rotate,
    VFlip,
    VStack,
)


def select_emoji_path():
    emoji_dir_all = glob.glob("AugLy/augly/assets/twemojis/*")
    emoji_dir = random.choice(emoji_dir_all)
    emoji_pic_all = glob.glob(f"{emoji_dir}/*.png")
    emoji_pic = random.choice(emoji_pic_all)
    return emoji_pic


def decord_clip_reader(path: Path, start: float, duration: float, count: int):
    num_threads = 0
    ctx = decord.cpu(0)

    reader = decord.VideoReader(
        path.as_posix(),
        ctx=ctx,
        num_threads=num_threads,
    )

    n_frames = len(reader)
    timestamps = reader.get_frame_timestamp(np.arange(n_frames))

    frame_pos = np.linspace(start, start + duration, count)
    frame_ids = np.searchsorted(timestamps[:, 0], frame_pos)
    frame_ids = np.minimum(frame_ids, n_frames - 1)

    frames = reader.get_batch(frame_ids).asnumpy()

    tensor = torch.as_tensor(frames)
    tensor = torch.permute(tensor, (0, 3, 1, 2))
    return tensor


def main(args):

    gt_match_csv = f"{args.output}/cvl_train_matching_ground_truth.csv"
    edit_query_meta_csv = f"{args.output}/cvl_train_query_metadata.csv"
    if not Path(gt_match_csv).is_file():
        os.makedirs(f"{args.output}/query", exist_ok=True)
        os.makedirs(f"{args.output}/cvl_train_augly_metadata", exist_ok=True)
        with open(gt_match_csv, "a") as f_gt:
            writer_gt = csv.writer(f_gt, delimiter=",")
            writer_gt.writerow(
                [
                    "query_id",
                    "ref_id",
                    "query_start",
                    "query_end",
                    "ref_start",
                    "ref_end",
                ]
            )
        with open(edit_query_meta_csv, "a") as f_meta:
            writer_meta = csv.writer(f_meta, delimiter=",")
            writer_meta.writerow(
                [
                    "video_id",
                    "duration_sec",
                    "frames_per_sec",
                    "width",
                    "height",
                    "rn",
                    "base_query_id",
                ]
            )

    ref_meta = pd.read_csv(args.meta_path)
    # for i in range(0, 1):
    for i in range(args.sample_range[0], args.sample_range[1]):
        ref_id = random.choice(list(Path(args.data_path).glob("*.mp4")))
        r_meta = ref_meta[ref_meta["video_id"] == ref_id.stem]
        # select r_start, r_end from r_meta (>5s, mean 12 s, Max: 54, Min: 2.4)
        r_dura_org = float(r_meta["duration_sec"])
        if r_dura_org > 5:
            r_dura_copy = random.uniform(4, min(r_dura_org - 1, 60))
            r_start = random.uniform(0, r_dura_org - r_dura_copy)
            r_end = r_start + r_dura_copy
            # extract r_start, r_end from ref video -> decord_clip_reader
            count = int(r_meta["frames_per_sec"] * r_dura_copy)
            copy_ref = decord_clip_reader(
                path=ref_id, start=r_start, duration=r_dura_copy, count=count
            )
        else:
            continue

        query_id = random.choice(list(Path(args.data_path).glob("*.mp4")))
        q_meta = ref_meta[ref_meta["video_id"] == query_id.stem]
        # extract 0~end from query video
        q_dura_org = float(q_meta["duration_sec"])
        count = int(q_meta["frames_per_sec"] * q_dura_org)
        query_all = decord_clip_reader(
            path=query_id, start=0.0, duration=q_dura_org, count=count
        )
        # resize to fit query video
        if copy_ref.shape[2:] != query_all.shape[2:]:
            transform = T.Resize((query_all.shape[2], query_all.shape[3]))
            copy_ref = transform(copy_ref)
        copy_ref = torch.permute(copy_ref, (0, 2, 3, 1))
        query_all = torch.permute(query_all, (0, 2, 3, 1))

        # select q_start, q_end from q_meta
        q_insert_frame = random.choice(range(0, query_all.shape[0]))
        # insert
        frameNo = query_all.shape[0] + copy_ref.shape[0]
        copy_video = torch.zeros(frameNo, query_all.shape[1], query_all.shape[2], 3)
        copy_video[:q_insert_frame, :, :, :] = query_all[:q_insert_frame, :, :, :]
        q_insert_end = q_insert_frame + copy_ref.shape[0]
        copy_video[q_insert_end:, :, :, :] = query_all[q_insert_frame:, :, :, :]
        copy_video[q_insert_frame:q_insert_end, :, :, :] = copy_ref

        video_in2 = random.choice(list(Path(args.data_path).glob("*.mp4")))
        emoji_path = select_emoji_path()
        brightness_level = random.uniform(-0.6, 0.6)
        contrast_level = random.uniform(-2.0, 2.0)
        saturation_factor = random.uniform(0.0, 3.0)
        rotate_degrees = random.choice([i for i in range(-30, 30, 5)])
        # rotate_degrees = random.choice([-90, 90])
        crop_left = random.uniform(0, 0.3)
        crop_top = random.uniform(0, 0.3)
        crop_right = random.uniform(0.7, 1.0)
        crop_bottom = random.uniform(0.7, 1.0)
        pad_w_factor = random.uniform(0, 0.25)
        pad_h_factor = random.uniform(0, 0.25)
        pad_color_r = random.randint(0, 255)
        pad_color_g = random.randint(0, 255)
        pad_color_b = random.randint(0, 255)
        overlay_num_dots = random.randint(50, 250)
        overlay_dot_type = random.choice(["colored", "blur"])
        overlay_random_movement = random.choice([True, False])
        overlay_num_shapes = random.randint(1, 3)
        overlay_text_len = random.randint(5, 15)
        shake_sigma = random.uniform(10, 50)
        shake_radius = random.uniform(0, 1.0)
        emoji_x_factor = random.uniform(0.1, 0.6)
        emoji_y_factor = random.uniform(0.1, 0.6)
        emoji_opacity = random.uniform(0.5, 1.0)
        emoji_size = random.uniform(0.2, 0.6)
        opacity = random.uniform(0.1, 0.5)
        overlay_size = random.choice([1.0, 1.0, 1.0, 0.8, 0.9])
        overlay_xy = [[0.2, 0], [0.3, 0], [0.4, 0], [0.5, 0]]
        overlay_xy_i = random.randint(0, len(overlay_xy) - 1)
        overlay_x_factor = overlay_xy[overlay_xy_i][0]
        overlay_y_factor = overlay_xy[overlay_xy_i][1]
        augly_method = [
            Brightness(level=brightness_level),
            Contrast(level=contrast_level),
            ColorJitter(saturation_factor=2.5),
            Grayscale(),
            RandomBlur(min_sigma=4.0, max_sigma=12.0),
            RandomPixelization(min_ratio=0.1, max_ratio=0.7),
            RandomAspectRatio(min_ratio=0.5, max_ratio=2.0),
            RandomVideoSpeed(min_factor=0.2, max_factor=5.0),
            Rotate(degrees=rotate_degrees),
            Crop(left=crop_left, top=crop_top, right=crop_right, bottom=crop_bottom),
            Pad(
                w_factor=pad_w_factor,
                h_factor=pad_w_factor,
                color=(pad_color_r, pad_color_g, pad_color_b),
            ),
            RandomNoise(min_level=15, max_level=60),
            RandomEncodingQuality(min_quality=10, max_quality=45),
            HFlip(),
            VFlip(),
            Loop(),
            OverlayDots(
                num_dots=overlay_num_dots,
                dot_type=overlay_dot_type,
                random_movement=overlay_random_movement,
            ),
            OverlayShapes(num_shapes=overlay_num_shapes),
            OverlayText(text_len=overlay_text_len),
            PerspectiveTransformAndShake(sigma=shake_sigma, shake_radius=shake_radius),
            OverlayEmoji(
                emoji_path=emoji_path,
                x_factor=emoji_x_factor,
                y_factor=emoji_y_factor,
                opacity=emoji_opacity,
                emoji_size=emoji_size,
            ),
            BlendVideos(str(video_in2), opacity=opacity, overlay_size=overlay_size),
            HStack(str(video_in2)),
            VStack(str(video_in2)),
            Overlay(
                str(video_in2), x_factor=overlay_x_factor, y_factor=overlay_y_factor
            ),
            InsertInBackground(str(video_in2), offset_factor=0.1),
        ]
        edit_method = random.choice(augly_method)

        # save video
        meta_list = []
        out_vid_path = f"{args.output}/query/Q4{str(i).zfill(5)}.mp4"
        out_meta_path = (
            f"{args.output}/cvl_train_augly_metadata/Q4{str(i).zfill(5)}.json"
        )
        with tempfile.TemporaryDirectory() as dir:
            copy_vid_path = f"{dir}/copy_video.mp4"
            torchvision.io.write_video(
                filename=copy_vid_path,
                video_array=copy_video,
                fps=float(q_meta["frames_per_sec"]),
            )
            edit_method(copy_vid_path, out_vid_path, metadata=meta_list)

        factor = 1
        if meta_list[0]["name"] == "change_video_speed":
            factor = meta_list[0]["factor"]
        with open(gt_match_csv, "a") as f_gt:
            writer_gt = csv.writer(f_gt, delimiter=",")
            q_start_sec = q_insert_frame / float(q_meta["frames_per_sec"]) / factor
            q_end_sec = (
                q_start_sec
                + copy_ref.shape[0] / float(q_meta["frames_per_sec"]) / factor
            )
            row = [
                f"Q4{str(i).zfill(5)}",
                ref_id.stem,
                q_start_sec,
                q_end_sec,
                r_start,
                r_end,
            ]
            writer_gt.writerow(row)

        with open(edit_query_meta_csv, "a") as f_meta:
            writer_meta = csv.writer(f_meta, delimiter=",")
            duration_sec = frameNo / float(q_meta["frames_per_sec"]) / factor
            row = [
                f"Q4{str(i).zfill(5)}",
                duration_sec,
                float(q_meta["frames_per_sec"]),
                int(q_meta["width"]),
                int(q_meta["height"]),
                i,
                query_id.stem,
            ]
            writer_meta.writerow(row)

        with open(out_meta_path, "w") as f:
            json.dump(meta_list[0], f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="competition_data/train/reference/"
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default="competition_data/train/train_reference_metadata.csv",
    )
    parser.add_argument("--output", type=str, default="gen_data/train_reference")
    parser.add_argument("--sample_range", nargs="*", type=int, required=True)
    args = parser.parse_args()
    main(args)
