"""Dump logits + Grad-CAM (7x7, predicted class) of ONNX models on cached image sets, for the follow-up analyses."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CROSS = "results/paper_pc/cross_dataset"
SIZE = 224


def image_dir(dataset: str, out: Path) -> Path:
    if dataset.startswith("paddy_"):
        return out / "_images" / dataset
    return Path(CROSS) / dataset


def load_images(dataset: str, out: Path):
    d = image_dir(dataset, out)
    meta = pd.read_csv(d / "images.csv", keep_default_na=False)
    return meta, np.load(d / "images_224.npy", mmap_mode="r")


def load_dump(out, dataset, run, variant):
    f = Path(out) / dataset / Path(run).name / f"{variant}.npz"
    return dict(np.load(f)) if f.exists() else None


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def mcnemar_p(ca, cb):
    from scipy.stats import binomtest

    b, c = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    return 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)


def prep(a):
    from PIL import Image

    from joiv.data import load_split

    out = Path(a.out)
    for sub in ("val", "test"):
        df = load_split(a.split, sub)[["path", "label"]]
        d = out / "_images" / f"paddy_{sub}"
        d.mkdir(parents=True, exist_ok=True)
        imgs = np.empty((len(df), SIZE, SIZE, 3), dtype=np.uint8)
        for i, p in enumerate(df.path):
            with Image.open(p) as im:
                imgs[i] = np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR))
        np.save(d / "images_224.npy", imgs)
        df.to_csv(d / "images.csv", index=False)
        print(d, len(df))


def dump(a):
    import onnxruntime as ort

    from joiv.model import cam_from_features
    from joiv.xai_metrics import normalise, stability, upsample

    out, run = Path(a.out), Path(a.run)
    head = json.loads((run / "onnx" / "head.json").read_text())
    classes, w = head["classes"], np.array(head["fc_weight"], dtype=np.float32)
    mean, std = np.array(head["mean"], np.float32), np.array(head["std"], np.float32)
    variants = []
    for v in a.variants:
        name, _, path = v.partition("=")
        variants.append((name, Path(path) if path else run / "onnx" / f"{name}.onnx"))
    variants.sort(key=lambda t: t[0] != "fp32")
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    for ds in a.datasets:
        meta, imgs = load_images(ds, out)
        od = out / ds / run.name
        od.mkdir(parents=True, exist_ok=True)
        for name, path in variants:
            f = od / f"{name}.npz"
            if f.exists() and not a.force:
                continue
            if not path.exists():
                print(f"missing {path}, skipped")
                continue
            t0 = time.time()
            s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
            n = len(meta)
            logits = np.zeros((n, len(classes)), np.float32)
            cams = np.zeros((n, 7, 7), np.float32)
            for i in range(n):
                x = ((imgs[i].astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)[None]
                lg, ft = s.run(None, {"image": np.ascontiguousarray(x, dtype=np.float32)})
                logits[i] = lg[0]
                cams[i] = cam_from_features(ft, w, [int(lg.argmax())])[0]
            res = {"logits": logits, "cam": cams}
            if name != "fp32":
                ref = load_dump(out, ds, run, "fp32")
                if ref is None:
                    raise SystemExit(f"fp32 dump missing for {ds} {run.name}")
                st = [stability(normalise(upsample(ref["cam"][i])), normalise(upsample(cams[i])), fracs=(0.10, 0.20))
                      for i in range(n)]
                for k in ("spearman", "ssim", "iou_top10", "iou_top20"):
                    res[k] = np.array([d[k] for d in st], np.float32)
            np.savez(f, **res)
            y = np.array([classes.index(l) if l in classes else -1 for l in meta.label])
            print(f"{ds} {run.name} {name}: acc10 {np.mean(logits.argmax(1) == y):.4f}  {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep")
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    d = sp.add_parser("dump")
    d.add_argument("--run", required=True)
    d.add_argument("--variants", nargs="+", default=["fp32", "int8_dyn", "int8_qdq", "int8_qdq_pct"])
    d.add_argument("--datasets", nargs="+", default=["paddy_val", "paddy_test", "sethy", "riceleafbd"])
    d.add_argument("--out", required=True)
    d.add_argument("--threads", type=int, default=4)
    d.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"prep": prep, "dump": dump}[a.cmd](a)


if __name__ == "__main__":
    main()
