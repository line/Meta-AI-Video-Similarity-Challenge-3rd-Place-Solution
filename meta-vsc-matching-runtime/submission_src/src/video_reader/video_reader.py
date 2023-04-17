# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Iterable, Optional, Tuple, Union

import torch
from PIL import Image

ImageT = Union[Image.Image, torch.Tensor]


class VideoReader(ABC):
    def __init__(self, video_path: str, required_fps: float, output_type: str) -> None:
        self.video_path = video_path
        self.required_fps = required_fps
        self.output_type = output_type
        self.original_fps = max(1, self.fps) if self.fps else 1
        self.video_frames = None

    @property
    @abstractmethod
    def fps(self) -> Optional[float]:
        pass

    @abstractmethod
    def frames(self) -> Iterable[Tuple[float, float, ImageT]]:
        """
        returns a tuple of [start_time, end_time, Image]
        """
        pass
