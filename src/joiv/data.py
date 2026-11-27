"""Dataset scanning, de-duplication, splitting and loading."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMG_SIZE = 224


def scan_class_folders(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMG_EXT and p.is_file():
            rows.append({"path": str(p.resolve()), "label": p.parent.name})
    if not rows:
        raise SystemExit(f"No images found under {root}")
    return pd.DataFrame(rows)


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def add_hashes(df: pd.DataFrame) -> pd.DataFrame:
    import imagehash

    md5s, phs = [], []
    for p in df["path"]:
        md5s.append(file_md5(p))
        with Image.open(p) as im:
            phs.append(str(imagehash.phash(im.convert("RGB"))))
    df = df.copy()
    df["md5"], df["phash"] = md5s, phs
    return df


def dedup(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    report = []
    df = df.drop_duplicates("md5", keep="first")
    kept_idx = []
    for ph, g in df.groupby("phash", sort=False):
        if len(g) == 1:
            kept_idx.append(g.index[0])
            continue
        if g["label"].nunique() > 1:
            report += [{"path": p, "label": l, "phash": ph, "action": "dropped_conflicting_labels"}
                       for p, l in zip(g["path"], g["label"])]
            continue
        kept_idx.append(g.index[0])
        report += [{"path": p, "label": l, "phash": ph, "action": "dropped_duplicate"}
                   for p, l in zip(g["path"].iloc[1:], g["label"].iloc[1:])]
    return df.loc[sorted(kept_idx)].reset_index(drop=True), pd.DataFrame(report)


def stratified_split(df: pd.DataFrame, val=0.15, test=0.15, seed=42) -> pd.DataFrame:
    from sklearn.model_selection import train_test_split

    rest, te = train_test_split(df, test_size=test, stratify=df["label"], random_state=seed)
    tr, va = train_test_split(rest, test_size=val / (1 - test), stratify=rest["label"], random_state=seed)
    out = pd.concat([tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")])
    return out.reset_index(drop=True)


def eval_transform(mean, std):
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(mean, std)])


def train_transform(mean, std):
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.6, 1.4)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(0.3, 0.3, 0.3, 0.03),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


class ImageTable(Dataset):
    def __init__(self, df: pd.DataFrame, classes: list[str], tf):
        self.paths = df["path"].tolist()
        self.y = [classes.index(l) for l in df["label"]]
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = self.tf(im.convert("RGB"))
        return x, torch.tensor(self.y[i])


def load_split(csv: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(csv)
    return df[df["split"] == split].reset_index(drop=True)


def preprocess_numpy(path: str, mean, std) -> np.ndarray:
    """Same preprocessing as eval_transform, in numpy, shape (1,3,H,W) float32."""
    with Image.open(path) as im:
        im = im.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = np.asarray(im, dtype=np.float32) / 255.0
    x = (x - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return x.transpose(2, 0, 1)[None]
