import torch
import torch.nn as nn
import torchvision


class TTA30ViewsTransform(nn.Module):
    """hstack/vstack, 回転（90, 180, 270度）, 左右反転, 上下反転の加工用TTA"""
    def __init__(self, base_transforms=None):
        super().__init__()
        self.base_transforms = base_transforms

    def forward(self, x) -> torch.Tensor:
        *_, h, w = x.shape

        x_top = x[..., :h // 2, :]  # top
        x_bottom = x[..., h // 2:, :]  # bottom
        x_left = x[..., :, :w // 2]  # left
        x_right = x[..., :, w // 2:]  # right

        if self.base_transforms is not None:
            x = self.base_transforms(x)
            x_top = self.base_transforms(x_top)
            x_bottom = self.base_transforms(x_bottom)
            x_left = self.base_transforms(x_left)
            x_right = self.base_transforms(x_right)

        crops = [
            x,
            torchvision.transforms.functional.rotate(x, angle=90),
            torchvision.transforms.functional.rotate(x, angle=180),
            torchvision.transforms.functional.rotate(x, angle=270),
            torchvision.transforms.functional.hflip(x),
            torchvision.transforms.functional.vflip(x),
            x_top,
            torchvision.transforms.functional.rotate(x_top, angle=90),
            torchvision.transforms.functional.rotate(x_top, angle=180),
            torchvision.transforms.functional.rotate(x_top, angle=270),
            torchvision.transforms.functional.hflip(x_top),
            torchvision.transforms.functional.vflip(x_top),
            x_bottom,
            torchvision.transforms.functional.rotate(x_bottom, angle=90),
            torchvision.transforms.functional.rotate(x_bottom, angle=180),
            torchvision.transforms.functional.rotate(x_bottom, angle=270),
            torchvision.transforms.functional.hflip(x_bottom),
            torchvision.transforms.functional.vflip(x_bottom),
            x_left,
            torchvision.transforms.functional.rotate(x_left, angle=90),
            torchvision.transforms.functional.rotate(x_left, angle=180),
            torchvision.transforms.functional.rotate(x_left, angle=270),
            torchvision.transforms.functional.hflip(x_left),
            torchvision.transforms.functional.vflip(x_left),
            x_right,
            torchvision.transforms.functional.rotate(x_right, angle=90),
            torchvision.transforms.functional.rotate(x_right, angle=180),
            torchvision.transforms.functional.rotate(x_right, angle=270),
            torchvision.transforms.functional.hflip(x_right),
            torchvision.transforms.functional.vflip(x_right),
        ]
        crops = torch.cat(crops, dim=0)

        return crops


class TTA24ViewsTransform(nn.Module):
    """hstack + vstack の加工用TTA"""
    def __init__(self, base_transforms=None):
        super().__init__()
        self.base_transforms = base_transforms

    def forward(self, x) -> torch.Tensor:
        *_, h, w = x.shape

        x_top_left = x[..., :h // 2, :w // 2]  # top_left
        x_top_right = x[..., :h // 2, w // 2:]  # top_right
        x_bottom_left = x[..., h // 2:, :w // 2]  # bottom_left
        x_bottom_right = x[..., h // 2:, w // 2:]  # bottom_right

        if self.base_transforms is not None:
            x_top_left = self.base_transforms(x_top_left)
            x_top_right = self.base_transforms(x_top_right)
            x_bottom_left = self.base_transforms(x_bottom_left)
            x_bottom_right = self.base_transforms(x_bottom_right)

        crops = [
            x_top_left,
            torchvision.transforms.functional.rotate(x_top_left, angle=90),
            torchvision.transforms.functional.rotate(x_top_left, angle=180),
            torchvision.transforms.functional.rotate(x_top_left, angle=270),
            torchvision.transforms.functional.hflip(x_top_left),
            torchvision.transforms.functional.vflip(x_top_left),
            x_top_right,
            torchvision.transforms.functional.rotate(x_top_right, angle=90),
            torchvision.transforms.functional.rotate(x_top_right, angle=180),
            torchvision.transforms.functional.rotate(x_top_right, angle=270),
            torchvision.transforms.functional.hflip(x_top_right),
            torchvision.transforms.functional.vflip(x_top_right),
            x_bottom_left,
            torchvision.transforms.functional.rotate(x_bottom_left, angle=90),
            torchvision.transforms.functional.rotate(x_bottom_left, angle=180),
            torchvision.transforms.functional.rotate(x_bottom_left, angle=270),
            torchvision.transforms.functional.hflip(x_bottom_left),
            torchvision.transforms.functional.vflip(x_bottom_left),
            x_bottom_right,
            torchvision.transforms.functional.rotate(x_bottom_right, angle=90),
            torchvision.transforms.functional.rotate(x_bottom_right, angle=180),
            torchvision.transforms.functional.rotate(x_bottom_right, angle=270),
            torchvision.transforms.functional.hflip(x_bottom_right),
            torchvision.transforms.functional.vflip(x_bottom_right),
        ]
        crops = torch.cat(crops, dim=0)

        return crops


class TTA5ViewsTransform(nn.Module):
    """左右反転, 上下反転の加工用TTA"""
    def __init__(self, base_transforms=None):
        super().__init__()
        self.base_transforms = base_transforms

    def forward(self, x) -> torch.Tensor:
        *_, h, w = x.shape

        x_top = x[..., :h // 2, :]  # top
        x_bottom = x[..., h // 2:, :]  # bottom
        x_left = x[..., :, :w // 2]  # left
        x_right = x[..., :, w // 2:]  # right

        if self.base_transforms is not None:
            x = self.base_transforms(x)
            x_top = self.base_transforms(x_top)
            x_bottom = self.base_transforms(x_bottom)
            x_left = self.base_transforms(x_left)
            x_right = self.base_transforms(x_right)

        crops = [
            x,
            x_top,
            x_bottom,
            x_left,
            x_right,
        ]
        crops = torch.cat(crops, dim=0)

        return crops


class TTA4ViewsTransform(nn.Module):
    """hstack + vstack の加工用TTA"""
    def __init__(self, base_transforms=None):
        super().__init__()
        self.base_transforms = base_transforms

    def forward(self, x) -> torch.Tensor:
        *_, h, w = x.shape

        x_top_left = x[..., :h // 2, :w // 2]  # top_left
        x_top_right = x[..., :h // 2, w // 2:]  # top_right
        x_bottom_left = x[..., h // 2:, :w // 2]  # bottom_left
        x_bottom_right = x[..., h // 2:, w // 2:]  # bottom_right

        if self.base_transforms is not None:
            x_top_left = self.base_transforms(x_top_left)
            x_top_right = self.base_transforms(x_top_right)
            x_bottom_left = self.base_transforms(x_bottom_left)
            x_bottom_right = self.base_transforms(x_bottom_right)

        crops = [
            x_top_left,
            x_top_right,
            x_bottom_left,
            x_bottom_right,
        ]
        crops = torch.cat(crops, dim=0)

        return crops


class TTAHorizontalStackTransform(nn.Module):
    def __init__(self, base_transforms=None):
        super().__init__()
        self.base_transforms = base_transforms

    def forward(self, x) -> torch.Tensor:
        *_, h, w = x.shape

        x_left = x[..., :, :w // 2]  # left
        x_right = x[..., :, w // 2:]  # right

        if self.base_transforms is not None:
            x_left = self.base_transforms(x_left)
            x_right = self.base_transforms(x_right)

        crops = [
            x_left,
            x_right,
        ]
        crops = torch.cat(crops, dim=0)

        return crops


class TTAVerticalStackTransform(nn.Module):
    """左右反転, 上下反転の加工用TTA"""
    def __init__(self, base_transforms=None):
        super().__init__()
        self.base_transforms = base_transforms

    def forward(self, x) -> torch.Tensor:
        *_, h, w = x.shape

        x_top = x[..., :h // 2, :]  # top
        x_bottom = x[..., h // 2:, :]  # bottom

        if self.base_transforms is not None:
            x_top = self.base_transforms(x_top)
            x_bottom = self.base_transforms(x_bottom)

        crops = [
            x_top,
            x_bottom,
        ]
        crops = torch.cat(crops, dim=0)

        return crops
