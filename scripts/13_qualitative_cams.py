"""Qualitative Grad-CAM figure: FP32 vs INT8-QDQ (MinMax) vs INT8-QDQ (Percentile)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PRECS = ["fp32", "int8_qdq", "int8_qdq_pct"]
PREC_LABEL = {"fp32": "FP32", "int8_qdq": "INT8-QDQ (MinMax)", "int8_qdq_pct": "INT8-QDQ-Pct"}
MODEL_LABEL = {"effb0": "EfficientNet-B0", "mnv2": "MobileNetV2", "mnv3l": "MobileNetV3-L", "effl0": "EfficientNet-Lite0"}
KIND_LABEL = {"flip": "MinMax flips", "drift": "MinMax CAM drifts", "stable": "stable"}


def pick_images(per_image: pd.DataFrame, n_flip: int, n_drift: int, n_stable: int) -> pd.DataFrame:
    w = per_image.pivot_table(index=["path", "label"], columns="precision", values="pred", aggfunc="first").reset_index()
    sp = per_image[per_image.precision == "int8_qdq"].set_index("path").spearman
    w["sp_qdq"] = w.path.map(sp)
    w = w[w.fp32 == w.label]
    used, out = set(), []

    def take(cand, n, kind):
        for _, r in cand.iterrows():
            if len([o for o in out if o["kind"] == kind]) == n:
                break
            if r.label in used:
                continue
            used.add(r.label)
            out.append({"path": r.path, "label": r.label, "kind": kind})

    take(w[(w.int8_qdq != w.fp32) & (w.int8_qdq_pct == w.fp32)].sort_values("sp_qdq"), n_flip, "flip")
    same = w[(w.int8_qdq == w.fp32) & (w.int8_qdq_pct == w.fp32)]
    take(same.sort_values("sp_qdq"), n_drift, "drift")
    take(same.sort_values("sp_qdq", ascending=False), n_stable, "stable")
    return pd.DataFrame(out)


def compute(a):
    import onnxruntime as ort
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from joiv.data import preprocess_numpy
    from joiv.model import cam_from_features

    runs = [Path(r) for r in a.runs]
    sel = pick_images(pd.read_csv(runs[0] / "eval" / "test_per_image.csv"), a.n_flip, a.n_drift, a.n_stable)
    print(sel.to_string())
    data = {"paths": sel.path.to_numpy(str), "labels": sel.label.to_numpy(str), "kinds": sel.kind.to_numpy(str),
            "runs": np.array([r.name for r in runs]), "precs": np.array(PRECS)}
    imgs = []
    for p in sel.path:
        with Image.open(p) as im:
            imgs.append(np.asarray(im.convert("RGB").resize((224, 224), Image.BILINEAR)))
    data["images"] = np.stack(imgs)
    for run in runs:
        head = json.loads((run / "onnx" / "head.json").read_text())
        classes, w = head["classes"], np.array(head["fc_weight"], dtype=np.float32)
        data[f"{run.name}__classes"] = np.array(classes)
        for prec in PRECS:
            s = ort.InferenceSession(str(run / "onnx" / f"{prec}.onnx"), providers=["CPUExecutionProvider"])
            cams, preds, confs = [], [], []
            for p in sel.path:
                lg, ft = s.run(None, {"image": preprocess_numpy(p, head["mean"], head["std"])})
                c = int(lg.argmax())
                z = np.exp(lg[0] - lg[0].max())
                cams.append(cam_from_features(ft, w, [c])[0])
                preds.append(c)
                confs.append(float(z[c] / z.sum()))
            data[f"{run.name}__{prec}__cam"] = np.stack(cams)
            data[f"{run.name}__{prec}__pred"] = np.array(preds)
            data[f"{run.name}__{prec}__conf"] = np.array(confs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **data)
    print("written", a.out)


def plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    d = np.load(a.npz)
    runs, kinds, labels, imgs = list(d["runs"]), d["kinds"], d["labels"], d["images"]
    n = len(labels)
    ncol = 1 + len(runs) * len(PRECS)
    fig, axes = plt.subplots(n, ncol, figsize=(1.32 * ncol, 1.42 * n + 0.45), squeeze=False)
    nice = lambda s: s.replace("_", " ")  # noqa: E731

    def up(cam):
        im = Image.fromarray(cam.astype(np.float32), mode="F").resize((224, 224), Image.BILINEAR)
        m = np.asarray(im, dtype=np.float32)
        lo, hi = m.min(), m.max()
        return (m - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(m)

    for i in range(n):
        ax = axes[i][0]
        ax.imshow(imgs[i])
        ax.set_ylabel(f"{nice(labels[i])}\n({KIND_LABEL[kinds[i]]})", fontsize=6.3, color="#0b0b0b")
        for j, run in enumerate(runs):
            classes = list(d[f"{run}__classes"])
            for k, prec in enumerate(PRECS):
                ax = axes[i][1 + j * len(PRECS) + k]
                ax.imshow(imgs[i])
                ax.imshow(up(d[f"{run}__{prec}__cam"][i]), cmap="inferno", alpha=0.55, vmin=0, vmax=1)
                c, cf = int(d[f"{run}__{prec}__pred"][i]), float(d[f"{run}__{prec}__conf"][i])
                ok = classes[c] == labels[i]
                ax.text(0.03, 0.03, f"{'' if ok else '✗ '}{nice(classes[c])} {cf:.2f}", transform=ax.transAxes,
                        fontsize=5.3, color="white", va="bottom", ha="left",
                        bbox=dict(boxstyle="round,pad=0.18", fc="#b3261e" if not ok else "#0b0b0b", ec="none", alpha=0.8))
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0][0].set_title("Input", fontsize=7)
    for j, run in enumerate(runs):
        for k, prec in enumerate(PRECS):
            axes[0][1 + j * len(PRECS) + k].set_title(PREC_LABEL[prec], fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.32 / (1.42 * n + 0.45)), h_pad=0.25, w_pad=0.15)
    for j, run in enumerate(runs):
        l = axes[0][1 + j * len(PRECS)].get_position()
        r = axes[0][(j + 1) * len(PRECS)].get_position()
        fig.text((l.x0 + r.x1) / 2, r.y1 + 0.032, MODEL_LABEL.get(run.split("_")[0], run) + f" (seed {run[-1]})",
                 ha="center", fontsize=7.5, weight="bold", color="#0b0b0b")
        if j:
            fig.add_artist(plt.Line2D([l.x0 - 0.006] * 2, [0.01, r.y1 + 0.03], color="#9a9994", lw=0.6))
    out = Path(a.out)
    for ext in ["pdf", "png"]:
        fig.savefig(out / f"fig_cams_qualitative.{ext}", dpi=300)
    plt.close(fig)
    print("written", out / "fig_cams_qualitative.pdf")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("compute")
    c.add_argument("--runs", nargs="+", required=True, help="first run decides the image choice")
    c.add_argument("--split", required=False)
    c.add_argument("--out", required=True)
    c.add_argument("--n-flip", type=int, default=3)
    c.add_argument("--n-drift", type=int, default=2)
    c.add_argument("--n-stable", type=int, default=1)
    p = sp.add_parser("plot")
    p.add_argument("--npz", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    compute(a) if a.cmd == "compute" else plot(a)


if __name__ == "__main__":
    main()
