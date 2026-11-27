"""Heatmap comparison metrics (stability between precisions, and localisation vs lesion masks)."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def upsample(cam: np.ndarray, size=(224, 224)) -> np.ndarray:
    from PIL import Image

    im = Image.fromarray(cam.astype(np.float32), mode="F").resize(size[::-1], Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def normalise(cam: np.ndarray) -> np.ndarray:
    lo, hi = float(cam.min()), float(cam.max())
    if hi - lo < 1e-12:
        return np.zeros_like(cam, dtype=np.float32)
    return ((cam - lo) / (hi - lo)).astype(np.float32)


def is_degenerate(cam: np.ndarray) -> bool:
    """Constant map (e.g. all zeros after ReLU)."""
    return float(cam.max() - cam.min()) < 1e-12


def topk_mask(cam: np.ndarray, frac: float) -> np.ndarray:
    flat = cam.ravel()
    k = max(1, int(round(frac * flat.size)))
    thr = np.partition(flat, -k)[-k]
    return cam >= thr


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else float("nan")


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    return float(structural_similarity(a, b, data_range=1.0))


def stability(ref: np.ndarray, other: np.ndarray, fracs=(0.05, 0.10, 0.15, 0.20, 0.30)) -> dict:
    """ref/other: normalised, same-size heatmaps."""
    out = {}
    if is_degenerate(ref) or is_degenerate(other):
        out["spearman"] = float("nan")
    else:
        out["spearman"] = float(spearmanr(ref.ravel(), other.ravel()).statistic)
    out["ssim"] = ssim(ref, other)
    for f in fracs:
        out[f"iou_top{int(f * 100)}"] = iou(topk_mask(ref, f), topk_mask(other, f))
    return out


def localisation(cam: np.ndarray, mask: np.ndarray, frac=0.15) -> dict:
    """cam: normalised heatmap; mask: bool lesion mask, same size."""
    total = cam.sum()
    return {
        "energy_pointing": float(cam[mask].sum() / total) if total > 0 else float("nan"),
        "pointing_hit": float(mask.ravel()[int(np.argmax(cam))]),
        f"iou_mask_top{int(frac * 100)}": iou(topk_mask(cam, frac), mask),
    }


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def deletion_insertion(x: np.ndarray, sal: np.ndarray, cls: int, predict, steps: int = 10, blur_sigma: float = 10.0) -> dict:
    """Faithfulness of a heatmap, no ground truth needed (Petsiuk et al., RISE, 2018)."""
    from scipy.ndimage import gaussian_filter

    h, w = sal.shape
    order = np.argsort(-sal.ravel(), kind="stable")
    blurred = np.stack([gaussian_filter(x[0, c], blur_sigma) for c in range(x.shape[1])])[None].astype(np.float32)
    fr = np.linspace(0, 1, steps + 1)
    dele, ins = [], []
    for f in fr:
        k = int(round(f * h * w))
        m = np.zeros(h * w, dtype=bool)
        m[order[:k]] = True
        m = m.reshape(1, 1, h, w)
        xd = np.where(m, 0.0, x).astype(np.float32)
        xi = np.where(m, x, blurred).astype(np.float32)
        dele.append(_softmax(predict(xd)[0])[cls])
        ins.append(_softmax(predict(xi)[0])[cls])
    trap = getattr(np, "trapezoid", None) or np.trapz
    return {"deletion_auc": float(trap(dele, fr)), "insertion_auc": float(trap(ins, fr))}
