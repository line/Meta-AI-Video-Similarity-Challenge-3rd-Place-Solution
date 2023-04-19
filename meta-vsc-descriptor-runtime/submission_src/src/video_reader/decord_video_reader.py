"""
Copyright 2023 LINE Corporation

LINE Corporation licenses this file to you under the Apache License,
version 2.0 (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.
"""
import os
from typing import Iterable, Optional, Tuple

import numpy as np
from PIL import Image
import decord
import torch
from torchvision.datasets.folder import default_loader

from src.video_reader.video_reader import VideoReader

ImageT = Image.Image


class DecordVideoReader(VideoReader):
    def __init__(self, video_path: str, required_fps: float, output_type: str):
        super().__init__(video_path, required_fps, output_type)

    @property
    def fps(self) -> Optional[float]:
        return self.required_fps

    def frames(self) -> Iterable[Tuple[float, float, ImageT]]:
        reader = decord.VideoReader(
            self.video_path,
            ctx=decord.cpu(0),
            num_threads=0,
        )

        n_frames = len(reader)
        timestamps = reader.get_frame_timestamp(np.arange(n_frames))
        end_timestamps = timestamps[:, 1]
        duration = end_timestamps[-1].tolist()

        fps = min(self.required_fps, n_frames / duration)
        count = max(1, np.round(duration * fps))
        step_size = 1 / fps

        frame_pos = (np.arange(count) + 0.5) * step_size

        frame_ids = np.searchsorted(end_timestamps, frame_pos)
        frame_ids = np.minimum(frame_ids, n_frames - 1)

        frames = reader.get_batch(frame_ids).asnumpy()

        for i, frame in enumerate(frames):
            if self.output_type == 'pil':
                img = Image.fromarray(frame).convert("RGB")
            elif self.output_type == 'tensor':
                img = torch.as_tensor(frame).permute(2, 0, 1)
            yield (
                i / self.original_fps,
                (i+1) / self.original_fps,
                img,
            )
