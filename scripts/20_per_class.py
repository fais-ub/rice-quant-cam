"""Per-class analysis of INT8 effects: flip rate and Grad-CAM stability vs FP32 for each disease class."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = {"mnv2": "MobileNetV2", "mnv3l": "MobileNetV3-L", "effl0": "EfficientNet-Lite0", "effb0": "EfficientNet-B0"}
PRECS = {"int8_qdq": "MinMax", "int8_qdq_pct": "Percentile"}
SHORT = {"bacterial_leaf_blight": "BLB", "bacterial_leaf_streak": "BLS", "bacterial_panicle_blight": "BPB",
         "blast": "Blast", "brown_spot": "Brown spot", "dead_heart": "Dead heart", "downy_mildew": "Downy mildew",
         "hispa": "Hispa", "normal": "Normal", "tungro": "Tungro"}


def per_class(pi, ref, dataset, run):
    rows = []
    ref = ref.set_index("key")
    for p, g in pi[pi.precision.isin(PRECS)].groupby("precision"):
        g = g.assign(ok32=g.key.map(ref.ok), pred32=g.key.map(ref.pred))
        for lab, gg in g.groupby("label"):
            rows.append({"dataset": dataset, "run": run, "precision": p, "label": lab, "n": len(gg),
                         "acc_fp32": float(gg.ok32.mean()), "acc": float((gg.pred == gg.label).mean()),
                         "flip": float((gg.pred != gg.pred32).mean()), "spearman": float(gg.spearman.mean()),
                         "iou_top20": float(gg.iou_top20.mean()), "spearman_same_pred":
                         float(gg[gg.pred == gg.pred32].spearman.mean())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for rd in sorted((data / "runs").glob("*_s[0-9]")):
        pi = pd.read_csv(rd / "eval" / "test_per_image.csv")
        pi["key"] = pi.path
        ref = pi[pi.precision == "fp32"].assign(ok=lambda d: d.pred == d.label)
        rows += per_class(pi, ref, "paddy_doctor", rd.name)
        for ds in ("sethy", "riceleafbd"):
            f = data / "results/paper_pc/cross_dataset" / ds / "runs" / f"{rd.name}_per_image.csv.gz"
            if f.exists():
                c = pd.read_csv(f, keep_default_na=False)
                c["key"] = c.idx
                c["spearman"] = pd.to_numeric(c.spearman, errors="coerce")
                c["iou_top20"] = pd.to_numeric(c.iou_top20, errors="coerce")
                r = c[c.precision == "fp32"].assign(ok=lambda d: d.pred == d.label)
                rows += per_class(c, r, ds, rd.name)
    L = pd.DataFrame(rows)
    L["model"] = L.run.str.split("_s").str[0]
    L["seed"] = L.run.str.split("_s").str[1].astype(int)
    L.to_csv(out / "per_class_long.csv", index=False)
    cols = ["acc_fp32", "acc", "flip", "spearman", "iou_top20", "spearman_same_pred"]
    g = L.groupby(["dataset", "model", "precision", "label"])
    S = g[cols].mean().add_suffix("_mean").join(g[cols].std().add_suffix("_sd")).join(g.n.first()).reset_index()
    S.to_csv(out / "per_class_summary.csv", index=False)

    D = S[S.dataset == "paddy_doctor"]
    classes = D.sort_values("n", ascending=False).label.unique()
    ms = list(MODELS)
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
             r"\caption{Per-class effect of static INT8 on the Paddy Doctor test set: top-1 flips vs FP32 (\%) and "
             r"Grad-CAM Spearman $\rho$ vs FP32, MinMax / Percentile calibration, mean over 3 seeds.}",
             r"\label{tab:perclass}", r"\begin{tabular}{lr" + "rrrr" * len(ms) + "}", r"\toprule",
             r" & & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{MODELS[m]}}}" for m in ms) + r" \\",
             "".join(rf"\cmidrule(lr){{{3 + 4 * i}-{6 + 4 * i}}}" for i in range(len(ms))),
             r" & & " + " & ".join([r"\multicolumn{2}{c}{Flip}", r"\multicolumn{2}{c}{$\rho$}"] * len(ms)) + r" \\",
             r"Class & $n$ & " + " & ".join(["MM", "Pct"] * 2 * len(ms)) + r" \\", r"\midrule"]
    for lab in classes:
        cells = []
        for m in ms:
            q = D[(D.model == m) & (D.label == lab)].set_index("precision")
            cells += [f"{100 * q.flip_mean.get(p, np.nan):.1f}" for p in PRECS]
            cells += [f"{q.spearman_mean.get(p, np.nan):.2f}" for p in PRECS]
        lines.append(f"{SHORT.get(lab, lab)} & {int(D[D.label == lab].n.iloc[0])} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (out / "tab_per_class.tex").write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 4, figsize=(10, 3.4), gridspec_kw={"wspace": 0.35})
    for k, (metric, lab, cmap, vmax) in enumerate([("flip_mean", "Top-1 flip vs FP32 (%)", "Oranges", None),
                                                   ("spearman_mean", "CAM Spearman vs FP32", "Blues_r", 1.0)]):
        for j, p in enumerate(PRECS):
            ax = axes[2 * k + j]
            M = np.array([[D[(D.model == m) & (D.label == c) & (D.precision == p)][metric].mean() for m in ms] for c in classes])
            if metric == "flip_mean":
                M = 100 * M
                vm = vmax or np.nanmax(100 * D[D.precision.isin(PRECS)].flip_mean)
                im = ax.imshow(M, cmap=cmap, vmin=0, vmax=vm, aspect="auto")
            else:
                im = ax.imshow(M, cmap=cmap, vmin=np.nanmin(D.spearman_mean), vmax=1, aspect="auto")
            for (i, jj), v in np.ndenumerate(M):
                ax.text(jj, i, f"{v:.0f}" if metric == "flip_mean" else f"{v:.2f}", ha="center", va="center", fontsize=5.5)
            ax.set_xticks(range(len(ms)), ["MNv2", "MNv3-L", "EffL0", "EffB0"], fontsize=6, rotation=30)
            ax.set_yticks(range(len(classes)), [SHORT[c] for c in classes] if j == 0 and k == 0 else [""] * len(classes),
                          fontsize=6.5)
            ax.set_title(f"{lab}\n{PRECS[p]}", fontsize=7)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(out / f"fig_per_class.{ext}", dpi=300)
    pd.set_option("display.width", 250)
    for p in PRECS:
        t = D[D.precision == p].pivot_table(index="label", columns="model", values=["flip_mean", "spearman_mean"])
        print(p, "\n", t.round(3).to_string())


if __name__ == "__main__":
    main()
