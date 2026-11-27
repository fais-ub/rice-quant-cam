"""Classifier with a GAP + single Linear head."""
from __future__ import annotations

import timm
import torch
from torch import nn

MODELS = {
    "mnv2": "mobilenetv2_100",
    "mnv3l": "mobilenetv3_large_100",
    "effl0": "tf_efficientnet_lite0",
    "effb0": "efficientnet_b0",
}


class CamNet(nn.Module):
    def __init__(self, arch: str, num_classes: int, pretrained: bool = True, drop: float = 0.2):
        super().__init__()
        name = MODELS.get(arch, arch)
        self.timm_name = name
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="")
        with torch.no_grad():
            c = self.backbone.forward_features(torch.zeros(1, 3, 224, 224)).shape[1]
        self.num_features = c
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(c, num_classes)
        cfg = timm.data.resolve_data_config({}, model=self.backbone)
        self.mean, self.std = tuple(cfg["mean"]), tuple(cfg["std"])

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def head(self, a: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(a.mean(dim=(2, 3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class ExportWrapper(nn.Module):
    """Two outputs for deployment: logits and the last feature map A."""

    def __init__(self, net: CamNet):
        super().__init__()
        self.net = net.eval()

    def forward(self, x):
        a = self.net.features(x)
        return self.net.head(a), a


def cam_from_features(a, w, cls):
    """Analytic Grad-CAM. a: (B,C,H,W) array/tensor, w: (K,C), cls: (B,) -> (B,H,W), unnormalised."""
    import numpy as np

    a = np.asarray(a, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    wc = w[np.asarray(cls)]
    cam = np.einsum("bchw,bc->bhw", a, wc) / (a.shape[2] * a.shape[3])
    return np.maximum(cam, 0.0)


def build_from_checkpoint(path: str, map_location="cpu") -> tuple[CamNet, dict]:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    net = CamNet(ck["arch"], len(ck["classes"]), pretrained=False)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck
