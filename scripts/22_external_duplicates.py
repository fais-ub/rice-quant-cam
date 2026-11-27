"""Near-duplicate groups in the external test sets and cross-dataset results on one image per group."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

PRECS = ["fp16", "int8_dyn", "int8_qdq", "int8_qdq_pct"]
MODELS = {"mnv2": "MobileNetV2", "mnv3l": "MobileNetV3-L", "effl0": "EfficientNet-Lite0", "effb0": "EfficientNet-B0"}
PREC = {"fp32": "FP32", "fp16": "FP16", "int8_dyn": "INT8 dyn.", "int8_qdq": "INT8 MinMax", "int8_qdq_pct": "INT8 Pct."}


def hashes(paths):
    import imagehash
    from PIL import Image, ImageOps

    rows = []
    for p in paths:
        with open(p, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        with Image.open(p) as im:
            im = im.convert("RGB")
            w, h = im.size
            ph = imagehash.phash(im)
            variants = []
            for k in range(4):
                r = im.rotate(90 * k, expand=True)
                variants += [imagehash.phash(r), imagehash.phash(ImageOps.mirror(r))]
        inv = min(int(str(v), 16) for v in variants)
        rows.append({"path": p, "md5": md5, "phash": str(ph), "phash_inv": f"{inv:016x}", "w": w, "h": h})
    return pd.DataFrame(rows)


def bits(hexes):
    v = np.array([int(h, 16) for h in hexes], dtype=np.uint64)
    return np.unpackbits(v.view(np.uint8).reshape(-1, 8), axis=1)


def groups(hexes, t):
    """Connected components of Hamming distance <= t; group id = smallest member index."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    b = bits(hexes).astype(np.int8)
    n = len(b)
    I, J = [], []
    for s in range(0, n, 256):
        d = (b[s:s + 256, None, :] != b[None, :, :]).sum(2)
        ii, jj = np.nonzero(d <= t)
        I.append(ii + s)
        J.append(jj)
    I, J = np.concatenate(I), np.concatenate(J)
    _, lab = connected_components(coo_matrix((np.ones(len(I), np.int8), (I, J)), shape=(n, n)), directed=False)
    first = pd.Series(np.arange(n)).groupby(lab).transform("min").to_numpy()
    return first


def metrics(c, keep):
    """Cross-dataset metrics from a 15_cross_dataset per-image table, restricted to idx in keep."""
    c = c[c.idx.isin(keep)]
    ref = c[c.precision == "fp32"].set_index("idx")
    rows = []
    for p, g in c.groupby("precision", sort=False):
        g = g.set_index("idx")
        r = {"precision": p, "n": len(g), "acc10": float((g.pred == g.label).mean()),
             "acc_ov": float((g.pred_ov == g.label).mean()), "flip": float((g.pred != ref.pred.reindex(g.index)).mean())}
        if p != "fp32":
            r["spearman"] = float(pd.to_numeric(g.spearman, errors="coerce").mean())
            r["iou_top20"] = float(pd.to_numeric(g.iou_top20, errors="coerce").mean())
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="+", default=["sethy", "riceleafbd"])
    ap.add_argument("--thresholds", nargs="+", type=int, default=[0, 4, 8])
    ap.add_argument("--dedup-threshold", type=int, default=4)
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cross = data / "results/paper_pc/cross_dataset"

    paddy = pd.read_csv(data / "splits/paddy.csv")
    pb = bits(paddy.phash)
    summ, long = [], []
    for ds in a.datasets:
        meta = pd.read_csv(cross / ds / "images.csv", keep_default_na=False)
        hf = out / f"{ds}_hashes.csv"
        if hf.exists():
            H = pd.read_csv(hf, dtype=str)
            H[["w", "h"]] = H[["w", "h"]].astype(int)
        else:
            H = hashes(meta.path.tolist())
            H.to_csv(hf, index=False)
        H["idx"] = np.arange(len(H))
        H["label"] = meta.label.values
        H["src_class"] = meta.src_class.values
        G = H[["idx", "path", "label", "md5", "phash", "phash_inv"]].copy()
        G["g_md5"] = G.groupby("md5").idx.transform("min")
        for t in a.thresholds:
            G[f"g_plain_t{t}"] = groups(G.phash, t)
            G[f"g_inv_t{t}"] = groups(G.phash_inv, t)
        G.to_csv(out / f"{ds}_groups.csv", index=False)
        row = {"dataset": ds, "n_images": len(G), "unique_md5": G.g_md5.nunique()}
        for t in a.thresholds:
            for kind in ("plain", "inv"):
                col = f"g_{kind}_t{t}"
                sz = G.groupby(col).size()
                row[f"groups_{kind}_t{t}"] = len(sz)
                row[f"dup_frac_{kind}_t{t}"] = 1 - len(sz) / len(G)
                row[f"max_group_{kind}_t{t}"] = int(sz.max())
                row[f"mixed_label_groups_{kind}_t{t}"] = int((G.groupby(col).label.nunique() > 1).sum())
        eb = bits(G.phash).astype(np.int16)
        mind = np.concatenate([(eb[s:s + 256, None, :] != pb[None].astype(np.int16)).sum(2).min(1)
                               for s in range(0, len(eb), 256)])
        G["min_dist_to_paddy"] = mind
        row["paddy_overlap_t0"] = int((mind == 0).sum())
        row["paddy_overlap_t4"] = int((mind <= 4).sum())
        summ.append(row)
        G.to_csv(out / f"{ds}_groups.csv", index=False)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

        gcol = f"g_inv_t{a.dedup_threshold}"
        keep_dedup = set(G.groupby(gcol).idx.min())
        keep_md5 = set(G.groupby("g_md5").idx.min())
        for f in sorted((cross / ds / "runs").glob("*_per_image.csv.gz")):
            run = f.name.replace("_per_image.csv.gz", "")
            c = pd.read_csv(f, keep_default_na=False)
            for subset, keep in (("all", set(G.idx)), ("dedup_md5", keep_md5), (f"dedup_inv_t{a.dedup_threshold}", keep_dedup)):
                for r in metrics(c, keep):
                    long.append({"dataset": ds, "run": run, "subset": subset, **r})
    S0 = pd.DataFrame(summ)
    S0.to_csv(out / "duplicate_summary.csv", index=False)
    L = pd.DataFrame(long)
    L["model"] = L.run.str.split("_s").str[0]
    L.to_csv(out / "cross_dedup_long.csv", index=False)
    cols = ["n", "acc10", "acc_ov", "flip", "spearman", "iou_top20"]
    g = L.groupby(["dataset", "subset", "model", "precision"], sort=False)
    S = g[cols].mean().add_suffix("_mean").join(g[cols].std().add_suffix("_sd")).reset_index()
    S.to_csv(out / "cross_dedup_summary.csv", index=False)

    sub = f"dedup_inv_t{a.dedup_threshold}"
    names = {"sethy": "Sethy et al.", "riceleafbd": "RiceLeafBD"}
    info = S0.set_index("dataset")
    cap = "; ".join(f"{names[d]}: {int(info.n_images[d])} images, {int(info[f'groups_inv_t{a.dedup_threshold}'][d])} "
                    f"after merging near-duplicates" for d in info.index)
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Cross-dataset results on all images vs one image per near-duplicate group (pHash Hamming "
             rf"$\le {a.dedup_threshold}$, invariant to flips and 90$^\circ$ rotations; {cap}). "
             r"Acc$_\mathrm{ov}$ of FP32, flip and Grad-CAM $\rho$ vs FP32, mean over 3 seeds.}",
             r"\label{tab:dedup}", r"\begin{tabular}{lll" + "rr" * 3 + "}", r"\toprule",
             r"Set & Model & Subset & \multicolumn{2}{c}{FP32 Acc$_\mathrm{ov}$ / MinMax flip} & "
             r"\multicolumn{2}{c}{Pct. flip / $\rho$} & \multicolumn{2}{c}{MinMax $\rho$ / $n$} \\", r"\midrule"]
    for ds in [d for d in ["sethy", "riceleafbd"] if d in set(S.dataset)]:
        for m in MODELS:
            for s_ in ("all", sub):
                q = S[(S.dataset == ds) & (S.model == m) & (S.subset == s_)].set_index("precision")
                lines.append(f"{names[ds]} & {MODELS[m]} & {'all' if s_ == 'all' else 'dedup'} & "
                             f"{100 * q.acc_ov_mean['fp32']:.1f} & {100 * q.flip_mean['int8_qdq']:.1f} & "
                             f"{100 * q.flip_mean['int8_qdq_pct']:.1f} & {q.spearman_mean['int8_qdq_pct']:.2f} & "
                             f"{q.spearman_mean['int8_qdq']:.2f} & {int(q.n_mean['fp32'])}" + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    (out / "tab_dedup.tex").write_text("\n".join(lines) + "\n")

    pd.set_option("display.width", 250)
    print(S0.T.to_string())
    print(S[S.precision.isin(["fp32", "int8_qdq", "int8_qdq_pct"])][["dataset", "subset", "model", "precision", "n_mean",
                                                                     "acc_ov_mean", "flip_mean", "spearman_mean"]]
          .round(3).to_string(index=False))


def figure(a):
    """Examples of the largest near-duplicate groups."""
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(a.out)
    fig, axes = plt.subplots(len(a.datasets) * 2, 6, figsize=(7, 2.4 * len(a.datasets)))
    for k, ds in enumerate(a.datasets):
        G = pd.read_csv(out / f"{ds}_groups.csv", keep_default_na=False)
        col = f"g_inv_t{a.dedup_threshold}"
        big = G.groupby(col).size().sort_values(ascending=False).index[:2]
        for r, gid in enumerate(big):
            mem = G[G[col] == gid].head(6)
            for j in range(6):
                ax = axes[2 * k + r][j]
                ax.axis("off")
                if j < len(mem):
                    with Image.open(mem.path.iloc[j]) as im:
                        ax.imshow(im.convert("RGB").resize((96, 96)))
                    ax.set_title(f"{ds} {mem.label.iloc[j][:10]}", fontsize=5)
    fig.tight_layout()
    fig.savefig(out / "fig_dup_examples.png", dpi=200)


if __name__ == "__main__":
    import sys

    if "--figure-only" in sys.argv:
        sys.argv.remove("--figure-only")
        ap = argparse.ArgumentParser()
        ap.add_argument("--data", required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("--datasets", nargs="+", default=["sethy", "riceleafbd"])
        ap.add_argument("--dedup-threshold", type=int, default=4)
        figure(ap.parse_args())
    else:
        main()
