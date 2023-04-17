# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Iterable, Optional, Tuple
import warnings

from PIL import Image
import ffmpeg
import numpy as np
import torch

from src.video_reader.video_reader import VideoReader

ImageT = Image.Image


class FFMpegPyVideoReader(VideoReader):
    def __init__(self, video_path: str, required_fps: float, output_type: str):
        super().__init__(video_path, required_fps, output_type)

    @property
    def fps(self) -> Optional[float]:
        return None

    def frames(self) -> Iterable[Tuple[float, float, ImageT]]:
        buffer, _ = (
            ffmpeg.input(self.video_path)
            .video.filter("fps", self.required_fps)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .run(capture_stdout=True, capture_stderr=True)
        )

        meta = ffmpeg.probe(self.video_path)["streams"][0]
        width, height = meta["width"], meta["height"]

        if self.output_type == 'pil':
            frames = np.frombuffer(buffer, dtype=np.uint8).reshape(-1, height, width, 3)
            for i, frame in enumerate(frames):
                img = Image.fromarray(frame).convert("RGB")
                yield i / self.original_fps, (i+1) / self.original_fps, img

        elif self.output_type == 'tensor':
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The given buffer is not writable")
                frames = torch.frombuffer(buffer, dtype=torch.uint8).reshape((-1, height, width, 3)).permute(0, 3, 1, 2)

            for i, frame in enumerate(frames):
                yield i / self.original_fps, (i+1) / self.original_fps, frame
