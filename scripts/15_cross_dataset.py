"""Cross-dataset test without retraining: Paddy Doctor models on external rice-leaf datasets."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PRECISIONS = ["fp32", "fp16", "int8_dyn", "int8_qdq", "int8_qdq_pct"]
MODELS = {"mnv2": "MobileNetV2", "mnv3l": "MobileNetV3-L", "effl0": "EfficientNet-Lite0", "effb0": "EfficientNet-B0"}
PREC = {"fp32": "FP32", "fp16": "FP16", "int8_dyn": "INT8 dynamic", "int8_qdq": "INT8 QDQ (MinMax)",
        "int8_qdq_pct": "INT8 QDQ (percentile)"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SIZE = 224


def prepare(a):
    from PIL import Image

    mp = pd.read_csv(a.mapping, keep_default_na=False)
    mp_d = mp[mp.dataset == a.dataset].set_index("src_class").paddy_class.to_dict()
    root = Path(a.root)
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            rows.append({"path": str(p), "src_class": p.parent.name})
    df = pd.DataFrame(rows)
    counts = df.src_class.value_counts().rename("n_images").reset_index()
    unknown = sorted(set(df.src_class) - set(mp_d))
    if unknown:
        raise SystemExit(f"classes missing from mapping for {a.dataset}: {unknown}")
    df["label"] = df.src_class.map(mp_d)
    use = df[df.label != ""].reset_index(drop=True)
    imgs = np.empty((len(use), SIZE, SIZE, 3), dtype=np.uint8)
    bad = []
    for i, p in enumerate(use.path):
        try:
            with Image.open(p) as im:
                imgs[i] = np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR))
        except Exception as e:  # noqa: BLE001
            bad.append((p, str(e)))
    if bad:
        keep = ~use.path.isin([b[0] for b in bad])
        imgs, use = imgs[keep.to_numpy()], use[keep].reset_index(drop=True)
        print(f"skipped {len(bad)} unreadable images, e.g. {bad[:3]}")
    out = Path(a.out) / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "images_224.npy", imgs)
    use.to_csv(out / "images.csv", index=False)
    counts.assign(dataset=a.dataset, paddy_class=counts.src_class.map(mp_d)).to_csv(out / "class_counts.csv", index=False)
    print(counts.assign(paddy_class=counts.src_class.map(mp_d)).to_string(index=False))
    print(f"{a.dataset}: {len(use)} of {len(df)} images used")


def mcnemar_p(ca, cb):
    from scipy.stats import binomtest

    b, c = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    return 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)


def eval_run(a, run):
    import onnxruntime as ort
    from sklearn.metrics import f1_score

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from joiv.model import cam_from_features
    from joiv.xai_metrics import normalise, stability, upsample

    d = Path(a.out) / a.dataset
    meta = pd.read_csv(d / "images.csv", keep_default_na=False)
    imgs = np.load(d / "images_224.npy", mmap_mode="r")
    head = json.loads((run / "onnx" / "head.json").read_text())
    classes, w = head["classes"], np.array(head["fc_weight"], dtype=np.float32)
    mean, std = np.array(head["mean"], dtype=np.float32), np.array(head["std"], dtype=np.float32)
    y = np.array([classes.index(l) for l in meta.label])
    ov = sorted(set(y))
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    sess = {p: ort.InferenceSession(str(run / "onnx" / f"{p}.onnx"), so, providers=["CPUExecutionProvider"])
            for p in PRECISIONS if (run / "onnx" / f"{p}.onnx").exists()}
    logits = {p: np.zeros((len(y), len(classes)), np.float32) for p in sess}
    cams = {p: [] for p in sess}
    t0 = time.time()
    for i in range(len(y)):
        x = ((imgs[i].astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
        for p, s in sess.items():
            lg, ft = s.run(None, {"image": x})
            logits[p][i] = lg[0]
            cams[p].append(cam_from_features(ft, w, [int(lg.argmax())])[0])
        if i % 500 == 0:
            print(f"{a.dataset} {run.name} {i}/{len(y)} {time.time() - t0:.0f}s", flush=True)

    pred = {p: logits[p].argmax(1) for p in sess}
    pred_ov = {p: np.array(ov)[logits[p][:, ov].argmax(1)] for p in sess}
    ok32 = pred["fp32"] == y
    rows, per = [], []
    for p in sess:
        rows.append({"dataset": a.dataset, "run": run.name, "precision": p, "n": len(y), "n_classes_overlap": len(ov),
                     "acc10": float(np.mean(pred[p] == y)),
                     "f1_10": float(f1_score(y, pred[p], labels=ov, average="macro", zero_division=0)),
                     "acc_ov": float(np.mean(pred_ov[p] == y)),
                     "f1_ov": float(f1_score(y, pred_ov[p], labels=ov, average="macro", zero_division=0)),
                     "flip": float(np.mean(pred[p] != pred["fp32"])),
                     "flip_ov": float(np.mean(pred_ov[p] != pred_ov["fp32"])),
                     "mcnemar_p": mcnemar_p(ok32, pred[p] == y),
                     "mcnemar_p_ov": mcnemar_p(pred_ov["fp32"] == y, pred_ov[p] == y),
                     "mean_maxprob": float(np.mean(1.0 / np.exp(logits[p] - logits[p].max(1, keepdims=True)).sum(1)))})
        for i in range(len(y)):
            rec = {"idx": i, "label": meta.label[i], "src_class": meta.src_class[i], "precision": p,
                   "pred": classes[pred[p][i]], "pred_ov": classes[pred_ov[p][i]], "same_pred": bool(pred[p][i] == pred["fp32"][i])}
            if p != "fp32":
                rec.update(stability(normalise(upsample(cams["fp32"][i])), normalise(upsample(cams[p][i])), fracs=(0.10, 0.20)))
            per.append(rec)
    pi = pd.DataFrame(per)
    cols = ["spearman", "ssim", "iou_top10", "iou_top20"]
    hm = []
    for p, g in pi[pi.precision != "fp32"].groupby("precision", sort=False):
        for scope, gg in (("all", g), ("same_pred", g[g.same_pred])):
            hm.append({"dataset": a.dataset, "run": run.name, "precision": p, "scope": scope, "n": len(gg),
                       **gg[cols].mean().to_dict()})
    od = d / "runs"
    od.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(od / f"{run.name}_classification.csv", index=False)
    pd.DataFrame(hm).to_csv(od / f"{run.name}_heatmap_summary.csv", index=False)
    pi.to_csv(od / f"{run.name}_per_image.csv.gz", index=False)
    print(pd.DataFrame(rows)[["precision", "acc10", "acc_ov", "f1_ov", "flip"]].round(3).to_string(index=False))
    print(f"done {a.dataset} {run.name} in {time.time() - t0:.0f}s", flush=True)


def evaluate(a):
    for r in a.runs:
        eval_run(a, Path(r))


def ms(s, d=3, scale=1.0):
    s = pd.Series(s).dropna() * scale
    if s.empty:
        return "--"
    if len(s) == 1:
        return f"{s.iloc[0]:.{d}f}"
    return f"{s.mean():.{d}f}{{\\scriptsize$\\pm${s.std():.{d}f}}}"


def summarize(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out, data = Path(a.out), Path(a.data)
    C, H = [], []
    for d in sorted(p for p in out.iterdir() if (p / "runs").is_dir()):
        for f in (d / "runs").glob("*_classification.csv"):
            C.append(pd.read_csv(f))
        for f in (d / "runs").glob("*_heatmap_summary.csv"):
            H.append(pd.read_csv(f))
    C, H = pd.concat(C, ignore_index=True), pd.concat(H, ignore_index=True)
    for t in (C, H):
        t["model"] = t.run.str.split("_s").str[0]
        t["seed"] = t.run.str.split("_s").str[1].astype(int)
    Hw = H.pivot_table(index=["dataset", "model", "seed", "precision"], columns="scope",
                       values=["spearman", "ssim", "iou_top10", "iou_top20", "n"]).reset_index()
    Hw.columns = [c if not s else f"{c}_{s}" for c, s in Hw.columns]
    L = C.merge(Hw, on=["dataset", "model", "seed", "precision"], how="left")

    R = []
    for m in MODELS:
        for rd in sorted((data / "runs").glob(f"{m}_s*")):
            c = pd.read_csv(rd / "eval" / "test_classification.csv")
            h = pd.read_csv(rd / "eval" / "test_heatmap_summary.csv")
            h = h[h.scope == "all"].set_index("precision")
            for _, r in c[c.precision.isin(PRECISIONS)].iterrows():
                R.append({"dataset": "paddy_doctor", "model": m, "seed": int(rd.name.split("_s")[1]), "precision": r.precision,
                          "n": r.n, "acc10": r.acc, "f1_10": r.macro_f1, "acc_ov": r.acc, "f1_ov": r.macro_f1,
                          "flip": r.flip_rate_vs_fp32, "mcnemar_p": r.mcnemar_p_vs_fp32,
                          "spearman_all": h.spearman.get(r.precision, np.nan), "ssim_all": h.ssim.get(r.precision, np.nan),
                          "iou_top20_all": h.iou_top20.get(r.precision, np.nan)})
    L = pd.concat([pd.DataFrame(R), L], ignore_index=True)
    L.to_csv(out / "cross_dataset_long.csv", index=False)
    cols = ["acc10", "f1_10", "acc_ov", "f1_ov", "flip", "mcnemar_p", "spearman_all", "ssim_all", "iou_top20_all",
            "spearman_same_pred", "iou_top20_same_pred"]
    g = L.groupby(["dataset", "model", "precision"], sort=False)
    S = g[cols].mean().add_suffix("_mean").join(g[cols].std().add_suffix("_sd"))
    S["n_seeds"] = g.size()
    S["mcnemar_n_sig05"] = g.mcnemar_p.apply(lambda s: int((s < 0.05).sum()))
    S = S.reset_index()
    S.to_csv(out / "cross_dataset_summary.csv", index=False)

    dsets = [d for d in L.dataset.unique() if d != "paddy_doctor"]
    names = {"paddy_doctor": "Paddy Doctor (in-domain)", "sethy": "Sethy et al.", "riceleafbd": "RiceLeafBD"}
    info = L.drop_duplicates("dataset").set_index("dataset")
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Cross-dataset test without retraining. Paddy Doctor models on external datasets, overlapping classes "
             r"only (" + "; ".join(f"{names.get(d, d)}: {int(info.n[d])} images, {int(info.n_classes_overlap[d])} classes"
                                  for d in dsets) + r"). Acc$_{10}$: argmax over all 10 classes; "
             r"Acc$_\mathrm{ov}$: argmax restricted to the overlapping classes. Flip, McNemar (seeds with $p<0.05$) and "
             r"Grad-CAM stability (Spearman $\rho$, IoU of the top 20\,\%) are against FP32 of the same run, all images. "
             rf"Mean $\pm$ sd over {L.seed.nunique()} seeds.}}",
             r"\label{tab:crossdataset}", r"\begin{tabular}{ll" + " rrrrr" * len(dsets) + "}", r"\toprule",
             "& & " + " & ".join(rf"\multicolumn{{5}}{{c}}{{{names.get(d, d)}}}" for d in dsets) + r" \\",
             "".join(rf"\cmidrule(lr){{{3 + 5 * i}-{7 + 5 * i}}}" for i in range(len(dsets))),
             "Model & Precision" + r" & Acc$_{10}$ & Acc$_\mathrm{ov}$ & Flip & $\rho$ & IoU$_{20}$" * len(dsets) + r" \\",
             r"\midrule"]
    for mi, m in enumerate(MODELS):
        for j, p in enumerate(PRECISIONS):
            cells = []
            for d in dsets:
                q = L[(L.dataset == d) & (L.model == m) & (L.precision == p)]
                cells += [ms(q.acc10, 1, 100), ms(q.acc_ov, 1, 100)]
                cells += ["--"] * 3 if p == "fp32" else [ms(q.flip, 1, 100), ms(q.spearman_all, 2), ms(q.iou_top20_all, 2)]
            lead = rf"\multirow{{{len(PRECISIONS)}}}{{*}}{{{MODELS[m]}}}" if j == 0 else ""
            lines.append(f"{lead} & {PREC[p]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule" if mi < len(MODELS) - 1 else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table*}"]
    (out / "tab_cross_dataset.tex").write_text("\n".join(lines) + "\n")

    MCOL = {"mnv2": "#2a78d6", "mnv3l": "#eb6834", "effl0": "#1baf7a", "effb0": "#eda100"}
    allsets = ["paddy_doctor"] + dsets
    precs = ["int8_dyn", "int8_qdq", "int8_qdq_pct"]
    short = {"int8_dyn": "INT8\ndyn", "int8_qdq": "INT8\nQDQ", "int8_qdq_pct": "INT8\nQDQ-Pct"}
    base = L[L.precision == "fp32"].set_index(["dataset", "model", "seed"]).acc_ov
    L["drop_ov"] = 100 * (L.set_index(["dataset", "model", "seed"]).index.map(base) - L.acc_ov)
    fig, axes = plt.subplots(2, len(allsets), figsize=(2.45 * len(allsets) + 0.4, 4.4), sharey="row", squeeze=False)
    w = 0.2
    for c, ds in enumerate(allsets):
        L["flip_pct"] = 100 * L.flip
        for r, (col, lab) in enumerate([("flip_pct", "Top-1 changed vs FP32 (%)"), ("spearman_all", "CAM Spearman vs FP32")]):
            ax = axes[r][c]
            for k, m in enumerate(MODELS):
                v = [L[(L.dataset == ds) & (L.model == m) & (L.precision == p)][col] for p in precs]
                xs = np.arange(len(precs)) + (k - 1.5) * w
                ax.bar(xs, [s.mean() for s in v], w - 0.03, yerr=[s.std() for s in v], color=MCOL[m],
                       error_kw=dict(lw=0.6, capsize=1.5, ecolor="#52514e"), label=MODELS[m])
            ax.set_xticks(range(len(precs)), [short[p] for p in precs] if r else [""] * len(precs), fontsize=7)
            if c == 0:
                ax.set_ylabel(lab, fontsize=7.5, color="#52514e")
            if r == 0:
                ax.set_title(names.get(ds, ds), fontsize=8, color="#0b0b0b")
            for s in ["top", "right"]:
                ax.spines[s].set_visible(False)
            ax.tick_params(colors="#52514e", labelsize=7)
            ax.grid(axis="y", color="#e4e3df", lw=0.6)
            ax.set_axisbelow(True)
            ax.axhline(0, color="#9a9994", lw=0.6)
    axes[1][0].set_ylim(0, 1.02)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ["pdf", "png"]:
        fig.savefig(out / f"fig_cross_dataset.{ext}", dpi=300)
    plt.close(fig)
    pd.set_option("display.width", 250)
    print(S[["dataset", "model", "precision", "acc10_mean", "acc_ov_mean", "flip_mean", "mcnemar_n_sig05",
             "spearman_all_mean", "iou_top20_all_mean"]].round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prepare")
    p.add_argument("--dataset", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--mapping", required=True)
    p.add_argument("--out", required=True)
    e = sp.add_parser("eval")
    e.add_argument("--dataset", required=True)
    e.add_argument("--runs", nargs="+", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--threads", type=int, default=4)
    s = sp.add_parser("summarize")
    s.add_argument("--out", required=True)
    s.add_argument("--data", required=True, help="joiv-data root (for the in-domain reference)")
    a = ap.parse_args()
    {"prepare": prepare, "eval": evaluate, "summarize": summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
