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
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class ISCNet(nn.Module):
    """
    Feature extractor for image copy-detection task.

    Args:
        backbone (`nn.Module`):
            Backbone module.
        fc_dim (`int=256`):
            Feature dimension of the fc layer.
        p (`float=1.0`):
            Power used in gem pooling for training.
        eval_p (`float=1.0`):
            Power used in gem pooling for evaluation. In practice, using a larger power
            for evaluation than training can yield a better performance.
    """

    def __init__(
        self,
        backbone: nn.Module,
        fc_dim: int = 256,
        p: float = 1.0,
        eval_p: float = 1.0,
        l2_normalize=True,
    ):
        super().__init__()

        self.backbone = backbone
        if hasattr(backbone, "num_features"):
            self.is_cnn = False
            in_channels = backbone.num_features
        else:
            self.is_cnn = True
            in_channels = backbone.feature_info.info[-1]["num_chs"]
        self.fc = nn.Linear(in_channels, fc_dim, bias=False)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        self.p = p
        self.eval_p = eval_p
        self.l2_normalize = l2_normalize

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_cnn:
            batch_size = x.shape[0]
            x = self.backbone(x)[-1]
            p = self.p if self.training else self.eval_p
            x = gem(x, p).view(batch_size, -1)
        else:
            x = self.backbone(x)
        x = self.fc(x)
        x = self.bn(x)
        if self.l2_normalize:
            x = F.normalize(x)
        return x


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class ConstDivider(nn.Module):
    def __init__(self, c=255.0):
        super().__init__()
        self.c = c

    def forward(self, x):
        return x / self.c


def create_model_in_runtime(transforms_device='cpu'):
    weight_path = "./model_assets/isc_ft_v107.pth.tar"
    ckpt = torch.load(weight_path)
    arch = ckpt["arch"]  # tf_efficientnetv2_m_in21ft1k
    input_size = ckpt["args"].input_size

    backbone = timm.create_model(arch, features_only=True)
    model = ISCNet(
        backbone=backbone,
        fc_dim=256,
        p=1.0,
        eval_p=1.0,
        l2_normalize=True,
    )
    model.to("cuda").train(False)

    state_dict = {}
    for s in ckpt["state_dict"]:
        state_dict[s.replace("module.", "")] = ckpt["state_dict"][s]

    model.load_state_dict(state_dict)

    if transforms_device == 'cpu':
        preprocessor = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                # transforms.ToTensor(),
                ConstDivider(c=255.0),
                transforms.Normalize(
                    mean=backbone.default_cfg["mean"],
                    std=backbone.default_cfg["std"],
                ),
            ]
        )
    else:
        preprocessor = nn.Sequential(
            transforms.Resize((input_size, input_size)),
            ConstDivider(c=255.0),
            transforms.Normalize(
                mean=backbone.default_cfg["mean"],
                std=backbone.default_cfg["std"],
            ),
        )
        preprocessor = torch.jit.script(preprocessor)
        preprocessor.to(transforms_device)

    return model, preprocessor


class CopyTypePredModel(nn.Module):
    def __init__(self, model_name, labels, transforms):
        super().__init__()
        self.model = timm.create_model(model_name, num_classes=len(labels))
        self.transforms = transforms
        self.labels = labels

    def forward(self, x):
        x = self.transforms(x)
        logit = self.model(x)
        proba = torch.sigmoid(logit)
        return {l: proba[:, i] for i, l in enumerate(self.labels)}


def create_copy_type_pred_model(transforms_device='cpu'):
    weight_path = "./model_assets/copy_type_pred__convnext_clip.ckpt"
    state_dict = torch.load(weight_path, map_location="cpu")["state_dict"]
    arch = "convnext_base_in22ft1k"
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    input_size = (256, 256)

    labels = [
        'add_noise',
        'blend_videos',
        'blur',
        'brightness',
        'change_aspect_ratio',
        'color_jitter',
        'contrast',
        'crop',
        'encoding_quality',
        'grayscale',
        'hstack',
        'other',
        'overlay',
        'overlay_dots',
        'overlay_emoji',
        'overlay_shapes',
        'overlay_text',
        'pad',
        'perspective_transform_and_shake',
        'pixelization',
        'rotate',
        'rotate180_vflip',
        'rotate90_270',
        'vstack',
        'vstack_hstack',
    ]

    if transforms_device == 'cpu':
        preprocessor = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=mean, std=std),
                transforms.Resize(size=input_size),
            ]
        )
    else:
        preprocessor = nn.Sequential(
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=mean, std=std),
            transforms.Resize(size=input_size),
        )
        preprocessor = torch.jit.script(preprocessor)
        preprocessor.to(transforms_device)

    model = CopyTypePredModel(arch, labels=labels, transforms=preprocessor)
    model.load_state_dict(state_dict)
    model.to("cuda").train(False)

    return model
