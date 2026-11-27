"""Sensitivity of static INT8 to the number of calibration images: MinMax vs Percentile, N = 32 / 100 / 300 / 1000."""
import argparse
import json
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SIZES = [32, 100, 300, 1000]
METHODS = ["minmax", "percentile"]
MODELS = {"mnv2": "MobileNetV2", "mnv3l": "MobileNetV3-L", "effl0": "EfficientNet-Lite0", "effb0": "EfficientNet-B0"}


def calib_lists(split):
    from joiv.data import load_split

    tr = load_split(split, "train")
    base = tr[tr.calib].path.tolist()
    rng = np.random.default_rng(0)
    perm = [base[i] for i in rng.permutation(len(base))]
    rest = tr[~tr.calib].path.tolist()
    extra = [rest[i] for i in rng.choice(len(rest), 700, replace=False)]
    return {32: perm[:32], 100: perm[:100], 300: base, 1000: base + extra}


def build(a):
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    CalibReader = import_module("03_export_onnx").CalibReader
    import_module("07_qdq_ablation")._chunk_histogram_calibration()
    run = Path(a.run)
    head = json.loads((run / "onnx" / "head.json").read_text())
    od = Path(a.out) / "models" / run.name
    od.mkdir(parents=True, exist_ok=True)
    lists = calib_lists(a.split)
    pd.DataFrame([{"n": n, "path": p} for n, ps in lists.items() for p in ps]).to_csv(od / "calib_lists.csv", index=False)
    pre = od / "_pre.onnx"
    quant_pre_process(str(run / "onnx" / "fp32.onnx"), str(pre))
    for m in a.methods:
        for n in a.sizes:
            f = od / f"{m}_n{n}.onnx"
            if f.exists() and not a.force:
                continue
            t0 = time.time()
            quantize_static(str(pre), str(f), CalibReader(lists[n], head["mean"], head["std"]),
                            quant_format=QuantFormat.QDQ, per_channel=True, reduce_range=False,
                            weight_type=QuantType.QInt8, activation_type=QuantType.QInt8,
                            calibrate_method=CalibrationMethod.MinMax if m == "minmax" else CalibrationMethod.Percentile)
            print(f"{run.name} {m} n={n}: {time.time() - t0:.0f}s", flush=True)
    pre.unlink(missing_ok=True)


def summarize(a):
    dump = import_module("16_dump_outputs")
    dumps, out = Path(a.dumps), Path(a.out)
    y_lab = pd.read_csv(dumps / "_images" / "paddy_test" / "images.csv").label.values
    rows = []
    for rd in sorted((dumps / "paddy_test").iterdir()):
        classes = json.loads(Path(f"runs/{rd.name}/onnx/head.json").read_text())["classes"]
        y = np.array([classes.index(l) for l in y_lab])
        ref = dump.load_dump(dumps, "paddy_test", rd.name, "fp32")
        if ref is None:
            continue
        p32 = ref["logits"].argmax(1)
        for m in METHODS:
            for n in SIZES:
                d = dump.load_dump(dumps, "paddy_test", rd.name, f"calib_{m}_n{n}")
                if d is None:
                    continue
                p = d["logits"].argmax(1)
                rows.append({"run": rd.name, "method": m, "n_calib": n, "acc": float(np.mean(p == y)),
                             "acc_fp32": float(np.mean(p32 == y)), "flip": float(np.mean(p != p32)),
                             "mcnemar_p": dump.mcnemar_p(p32 == y, p == y),
                             "spearman": float(np.nanmean(d["spearman"])), "iou_top20": float(np.nanmean(d["iou_top20"]))})
    L = pd.DataFrame(rows)
    L["model"] = L.run.str.split("_s").str[0]
    L["seed"] = L.run.str.split("_s").str[1].astype(int)
    L.to_csv(out / "calib_size_long.csv", index=False)
    cols = ["acc", "acc_fp32", "flip", "spearman", "iou_top20"]
    g = L.groupby(["model", "method", "n_calib"])
    S = g[cols].mean().add_suffix("_mean").join(g[cols].std().add_suffix("_sd"))
    S["n_seeds"] = g.size()
    S["mcnemar_n_sig05"] = g.mcnemar_p.apply(lambda s: int((s < 0.05).sum()))
    S = S.reset_index()
    S.to_csv(out / "calib_size_summary.csv", index=False)
    seeds = sorted(L.seed.unique())
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Number of calibration images vs static INT8 quality (Paddy Doctor test set; S8S8, per-channel "
             r"weights, full range). Accuracy (\%), top-1 flips vs FP32 (\%) and Grad-CAM Spearman $\rho$ vs FP32"
             + (f", mean over {len(seeds)} seeds" if len(seeds) > 1 else f", seed {seeds[0]}") + r".}",
             r"\label{tab:calibsize}", r"\begin{tabular}{llr" + "rrr" * 2 + "}", r"\toprule",
             r" & & & \multicolumn{3}{c}{MinMax} & \multicolumn{3}{c}{Percentile} \\",
             r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
             r"Model & FP32 acc & $N$ & Acc & Flip & $\rho$ & Acc & Flip & $\rho$ \\", r"\midrule"]
    for mi, mdl in enumerate([m for m in MODELS if m in set(S.model)]):
        for j, n in enumerate(SIZES):
            cells = []
            for m in METHODS:
                q = S[(S.model == mdl) & (S.method == m) & (S.n_calib == n)]
                cells += (["--"] * 3 if q.empty else
                          [f"{100 * q.acc_mean.iloc[0]:.1f}", f"{100 * q.flip_mean.iloc[0]:.1f}", f"{q.spearman_mean.iloc[0]:.3f}"])
            fa = S[S.model == mdl].acc_fp32_mean.iloc[0]
            lines.append((f"{MODELS[mdl]} & {100 * fa:.1f}" if j == 0 else " & ") + f" & {n} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    (out / "tab_calib_size.tex").write_text("\n".join(lines) + "\n")
    pd.set_option("display.width", 250)
    print(S.round(4).to_string(index=False))


def figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    S = pd.read_csv(out / "calib_size_summary.csv")
    MCOL = {"mnv2": "#2a78d6", "mnv3l": "#eb6834", "effl0": "#1baf7a", "effb0": "#eda100"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for k, (col, lab, sc) in enumerate([("acc", "Test accuracy (%)", 100), ("flip", "Top-1 changed vs FP32 (%)", 100),
                                        ("spearman", "CAM Spearman vs FP32", 1)]):
        ax = axes[k]
        for m in MODELS:
            for meth, ls in (("minmax", ":"), ("percentile", "-")):
                q = S[(S.model == m) & (S.method == meth)].sort_values("n_calib")
                if q.empty:
                    continue
                ax.errorbar(q.n_calib, sc * q[f"{col}_mean"], yerr=sc * q[f"{col}_sd"].fillna(0), ls=ls, marker="o", ms=2.5,
                            lw=1.1, color=MCOL[m], capsize=1.5, label=MODELS[m] if meth == "percentile" else None)
        ax.set_xscale("log")
        ax.set_xticks([32, 100, 300, 1000], ["32", "100", "300", "1000"])
        ax.set_xlabel("calibration images", fontsize=7)
        ax.set_ylabel(lab, fontsize=7)
        ax.tick_params(labelsize=6.5, colors="#52514e")
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axes[0].legend(fontsize=6, frameon=False, title="solid Pct., dotted MinMax", title_fontsize=6)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(out / f"fig_calib_size.{ext}", dpi=300)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--run", required=True)
    b.add_argument("--split", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--methods", nargs="+", default=METHODS)
    b.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    b.add_argument("--force", action="store_true", help="rebuild models that already exist")
    s = sp.add_parser("summarize")
    s.add_argument("--dumps", required=True)
    s.add_argument("--out", required=True)
    f = sp.add_parser("figure")
    f.add_argument("--out", required=True)
    a = ap.parse_args()
    {"build": build, "summarize": summarize, "figure": lambda a: figure(a.out)}[a.cmd](a)


if __name__ == "__main__":
    main()
