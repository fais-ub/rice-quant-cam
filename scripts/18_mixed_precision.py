"""Mixed-precision INT8 for the SE/swish models: INT8 Percentile with a few nodes kept in FP32 (nodes_to_exclude)."""
import argparse
import json
import re
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

MODELS = {"mnv2": "MobileNetV2", "mnv3l": "MobileNetV3-L", "effl0": "EfficientNet-Lite0", "effb0": "EfficientNet-B0"}
DEFAULT_LEVELS = ["pct", "se", "stem", "stem_se", "top1", "top2", "top4", "top8", "top16"]
DATASETS = ["paddy_val", "paddy_test", "sethy", "riceleafbd"]


def fpname(t):
    return t[: -len("_QuantizeLinear_Input")] if t.endswith("_QuantizeLinear_Input") else t


def node_sets(pre, level, rank_dir, arch):
    nodes = list(pre.graph.node)
    names = [n.name for n in nodes]
    producers = {o: n.name for n in nodes for o in n.output}
    consumers = {}
    for n in nodes:
        for i in n.input:
            consumers.setdefault(i, []).append(n.name)

    def stem():
        out = set()
        for n in names:
            if n.startswith("/blocks/"):
                break
            out.add(n)
        return out

    def se():
        return {n for n in names if "/se/" in n}

    if level == "pct":
        return set(), []
    if level == "se":
        return se(), []
    if level == "stem":
        return stem(), []
    if level == "stem_se":
        return stem() | se(), []
    m = re.fullmatch(r"top(\d+)", level)
    if m:
        k = int(m.group(1))
        s = pd.read_csv(Path(rank_dir) / f"{arch}_s0" / "sensitivity.csv")
        full = s[(s.qmodel == "pct") & (s.variant == "full_fake_quant")].kl.iloc[0]
        loo = s[(s.qmodel == "pct") & (s.variant == "leave_one_out")].copy()
        loo["gain"] = full - loo.kl
        tensors = [fpname(t) for t in loo.sort_values("gain", ascending=False).tensor.head(k)]
        ex = set()
        for t in tensors:
            if t in producers:
                ex.add(producers[t])
            ex.update(consumers.get(t, []))
        return ex, tensors
    raise SystemExit(f"unknown level {level}")


def build(a):
    import onnx
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    from joiv.data import load_split

    CalibReader = import_module("03_export_onnx").CalibReader
    import_module("07_qdq_ablation")._chunk_histogram_calibration()
    run = Path(a.run)
    arch = run.name.split("_s")[0]
    head = json.loads((run / "onnx" / "head.json").read_text())
    od = Path(a.out) / "models" / run.name
    od.mkdir(parents=True, exist_ok=True)
    pre = od / "_pre.onnx"
    if not pre.exists():
        quant_pre_process(str(run / "onnx" / "fp32.onnx"), str(pre))
    pm = onnx.load(str(pre))
    n_nodes = len(pm.graph.node)
    tr = load_split(a.split, "train")
    calib = tr[tr["calib"]]["path"].tolist()
    cache = od / "calib_percentile.json"
    info = []
    for level in a.levels:
        ex, tensors = node_sets(pm, level, a.rank, arch)
        f = od / f"mp_{level}.onnx"
        (od / f"nodes_{level}.txt").write_text("\n".join(sorted(ex)) + ("\n# tensors kept float:\n" + "\n".join(tensors)
                                                                       if tensors else "") + "\n")
        if not f.exists() or a.force:
            t0 = time.time()
            quantize_static(str(pre), str(f), CalibReader(calib, head["mean"], head["std"]) if not cache.exists() else None,
                            quant_format=QuantFormat.QDQ, per_channel=True, reduce_range=False,
                            weight_type=QuantType.QInt8, activation_type=QuantType.QInt8,
                            calibrate_method=CalibrationMethod.Percentile, nodes_to_exclude=sorted(ex),
                            calibration_cache_path=str(cache))
            print(f"{run.name} {level}: {len(ex)} nodes excluded, {time.time() - t0:.0f}s", flush=True)
        qm = onnx.load(str(f))
        info.append({"run": run.name, "level": level, "n_nodes": n_nodes, "n_excluded": len(ex),
                     "n_quantize_nodes": sum(n.op_type == "QuantizeLinear" for n in qm.graph.node),
                     "size_mb": f.stat().st_size / 1e6, "tensors_float": ";".join(tensors)})
    info = pd.DataFrame(info)
    if (od / "build_info.csv").exists():
        old = pd.read_csv(od / "build_info.csv", keep_default_na=False)
        info = pd.concat([old[~old.level.isin(info.level)], info], ignore_index=True)
    info.to_csv(od / "build_info.csv", index=False)
    print(info.drop(columns="tensors_float").to_string(index=False))


def latency(a):
    import onnxruntime as ort

    rows = []
    x = np.random.default_rng(0).standard_normal((1, 3, 224, 224)).astype(np.float32)
    for r in a.runs:
        run = Path(r)
        md = Path(a.out) / "models" / run.name
        files = [("fp32", run / "onnx" / "fp32.onnx"), ("int8_qdq_pct", run / "onnx" / "int8_qdq_pct.onnx")]
        files += [(f.stem, f) for f in sorted(md.glob("mp_*.onnx"))]
        for th in a.threads:
            for name, f in files:
                so = ort.SessionOptions()
                so.intra_op_num_threads = th
                s = ort.InferenceSession(str(f), so, providers=["CPUExecutionProvider"])
                for _ in range(20):
                    s.run(None, {"image": x})
                ts = []
                for _ in range(a.reps):
                    t0 = time.perf_counter()
                    s.run(None, {"image": x})
                    ts.append(1e3 * (time.perf_counter() - t0))
                rows.append({"run": run.name, "variant": name, "threads": th, "median_ms": float(np.median(ts)),
                             "p90_ms": float(np.percentile(ts, 90)), "size_mb": f.stat().st_size / 1e6})
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(Path(a.out) / "pc_latency.csv", index=False)


def summarize(a):
    dump = import_module("16_dump_outputs")
    dumps, out = Path(a.dumps), Path(a.out)
    rows = []
    variants = ["fp32", "int8_qdq", "int8_qdq_fr", "int8_qdq_pct"] + [f"mp_{l}" for l in a.levels]
    for run in a.runs:
        run = Path(run).name
        classes = json.loads(Path(f"runs/{run}/onnx/head.json").read_text())["classes"]
        binfo = pd.read_csv(out / "models" / run / "build_info.csv").set_index("level")
        for ds in DATASETS:
            meta = pd.read_csv((dumps / "_images" / ds if ds.startswith("paddy") else Path("results/paper_pc/cross_dataset") / ds)
                               / "images.csv", keep_default_na=False)
            y = np.array([classes.index(l) for l in meta.label])
            ov = sorted(set(y))
            ref = dump.load_dump(dumps, ds, run, "fp32")
            if ref is None:
                continue
            p32 = ref["logits"].argmax(1)
            for v in variants:
                d = dump.load_dump(dumps, ds, run, v)
                if d is None:
                    continue
                p = d["logits"].argmax(1)
                pov = np.array(ov)[d["logits"][:, ov].argmax(1)]
                r = {"run": run, "dataset": ds, "variant": v, "n": len(y), "acc10": float(np.mean(p == y)),
                     "acc_ov": float(np.mean(pov == y)), "flip": float(np.mean(p != p32)),
                     "mcnemar_p": dump.mcnemar_p(p32 == y, p == y)}
                if v != "fp32":
                    same = p == p32
                    r.update({"spearman": float(np.nanmean(d["spearman"])), "ssim": float(np.nanmean(d["ssim"])),
                              "iou_top20": float(np.nanmean(d["iou_top20"])),
                              "spearman_same_pred": float(np.nanmean(d["spearman"][same]))})
                lv = v[3:] if v.startswith("mp_") else None
                if lv in binfo.index:
                    r.update({"n_excluded": int(binfo.n_excluded[lv]), "n_nodes": int(binfo.n_nodes[lv]),
                              "size_mb": float(binfo.size_mb[lv])})
                rows.append(r)
    L = pd.DataFrame(rows)
    L["model"] = L.run.str.split("_s").str[0]
    L["seed"] = L.run.str.split("_s").str[1].astype(int)
    lat = out / "pc_latency.csv"
    if lat.exists():
        P = pd.read_csv(lat)
        P = P.pivot_table(index=["run", "variant"], columns="threads", values="median_ms")
        P.columns = [f"pc_ms_t{c}" for c in P.columns]
        L = L.merge(P.reset_index(), on=["run", "variant"], how="left")
    L.to_csv(out / "mp_long.csv", index=False)
    cols = [c for c in ["acc10", "acc_ov", "flip", "spearman", "ssim", "iou_top20", "spearman_same_pred", "n_excluded",
                        "size_mb", "pc_ms_t1", "pc_ms_t4"] if c in L]
    g = L.groupby(["dataset", "model", "variant"], sort=False)
    S = g[cols].mean().add_suffix("_mean").join(g[cols].std().add_suffix("_sd"))
    S["n_seeds"] = g.size()
    S["mcnemar_n_sig05"] = g.mcnemar_p.apply(lambda s: int((s < 0.05).sum()))
    S = S.reset_index()
    S.to_csv(out / "mp_summary.csv", index=False)

    def ms(q, c, d=1, scale=100.0):
        if q.empty or pd.isna(q[f"{c}_mean"].iloc[0]):
            return "--"
        m_, s_ = q[f"{c}_mean"].iloc[0] * scale, q[f"{c}_sd"].iloc[0] * scale
        return f"{m_:.{d}f}{{\\scriptsize$\\pm${s_:.{d}f}}}"

    name = {"fp32": "FP32", "int8_qdq": "INT8 MinMax", "int8_qdq_pct": "INT8 Pct.", **{f"mp_{l}": f"Pct. + FP32 {l}" for l in a.levels}}
    name.update({"mp_pct": "Pct. (rebuilt)", "mp_se": "Pct. + FP32 SE", "mp_stem": "Pct. + FP32 stem",
                 "mp_stem_se": "Pct. + FP32 stem, SE"})
    show = ["fp32", "int8_qdq", "int8_qdq_pct"] + [f"mp_{l}" for l in a.levels if l != "pct"]
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Mixed precision for the SE/swish models: INT8 Percentile with a few nodes left in FP32. "
             r"Paddy Doctor test set (accuracy, flips vs FP32, McNemar seeds with $p<0.05$, Grad-CAM Spearman $\rho$) "
             r"and the two external sets (flip and $\rho$ vs FP32). Nodes: FP32 nodes out of the graph's total. "
             r"top$k$: the $k$ activation tensors with the largest leave-one-out gain on seed-0 validation images. "
             r"PC latency: onnxruntime CPU, 1 thread, median. Mean $\pm$ sd over 3 seeds.}",
             r"\label{tab:mixedprec}", r"\begin{tabular}{llrrrrrrrrrr}", r"\toprule",
             r" & & & \multicolumn{4}{c}{Paddy Doctor test} & \multicolumn{2}{c}{Sethy et al.} & \multicolumn{2}{c}{RiceLeafBD} & \\",
             r"\cmidrule(lr){4-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
             r"Model & Variant & Nodes & Acc & Flip & McN. & $\rho$ & Flip & $\rho$ & Flip & $\rho$ & ms \\", r"\midrule"]
    for mi, m in enumerate([m for m in MODELS if m in set(S.model)]):
        for j, v in enumerate([v for v in show if not S[(S.model == m) & (S.variant == v)].empty]):
            t = S[(S.dataset == "paddy_test") & (S.model == m) & (S.variant == v)]
            cells = [("--" if v in ("fp32", "int8_qdq", "int8_qdq_pct") or t.empty else
                      f"{t.n_excluded_mean.iloc[0]:.0f}/{int(L[(L.model == m)].n_nodes.dropna().iloc[0])}"),
                     ms(t, "acc10"), "--" if v == "fp32" else ms(t, "flip"),
                     "--" if v == "fp32" else f"{int(t.mcnemar_n_sig05.iloc[0])}/{int(t.n_seeds.iloc[0])}",
                     "--" if v == "fp32" else ms(t, "spearman", 2, 1)]
            for ds in ["sethy", "riceleafbd"]:
                q = S[(S.dataset == ds) & (S.model == m) & (S.variant == v)]
                cells += ["--", "--"] if v == "fp32" else [ms(q, "flip"), ms(q, "spearman", 2, 1)]
            cells.append(ms(t, "pc_ms_t1", 1, 1) if "pc_ms_t1_mean" in t else "--")
            lines.append(f"{MODELS[m] if j == 0 else ''} & {name.get(v, v)} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}"]
    (out / "tab_mixed_precision.tex").write_text("\n".join(lines) + "\n")
    pd.set_option("display.width", 250)
    print(S[S.dataset.isin(["paddy_val", "paddy_test"])][["dataset", "model", "variant", "acc10_mean", "flip_mean",
                                                          "mcnemar_n_sig05", "spearman_mean", "n_excluded_mean"]]
          .round(4).to_string(index=False))
    print(S[~S.dataset.isin(["paddy_val", "paddy_test"])][["dataset", "model", "variant", "acc_ov_mean", "flip_mean",
                                                           "spearman_mean"]].round(3).to_string(index=False))


def figure(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(a.out)
    S = pd.read_csv(out / "mp_summary.csv")
    ms_ = [m for m in ["effb0", "mnv3l"] if m in set(S.model)]
    order = ["int8_qdq", "int8_qdq_pct"] + [f"mp_{l}" for l in a.levels if l != "pct"]
    lab = {"int8_qdq": "MinMax", "int8_qdq_pct": "Pct.", "mp_se": "+SE", "mp_stem": "+stem", "mp_stem_se": "+stem+SE"}
    COL = {"paddy_test": "#2a78d6", "sethy": "#eb6834", "riceleafbd": "#1baf7a"}
    NM = {"paddy_test": "Paddy Doctor test", "sethy": "Sethy et al.", "riceleafbd": "RiceLeafBD"}
    fig, axes = plt.subplots(2, len(ms_), figsize=(3.6 * len(ms_), 4.2), squeeze=False, sharex="col")
    for c, m in enumerate(ms_):
        vs = [v for v in order if not S[(S.model == m) & (S.variant == v)].empty]
        xs = np.arange(len(vs))
        for r, (col, yl, sc) in enumerate([("flip", "Top-1 changed vs FP32 (%)", 100), ("spearman", "CAM Spearman vs FP32", 1)]):
            ax = axes[r][c]
            for k, ds in enumerate(COL):
                q = S[(S.model == m) & (S.dataset == ds)].set_index("variant").reindex(vs)
                ax.errorbar(xs + (k - 1) * 0.12, sc * q[f"{col}_mean"], yerr=sc * q[f"{col}_sd"], fmt="o-", ms=3, lw=1,
                            capsize=1.5, color=COL[ds], label=NM[ds])
            ax.set_ylabel(yl, fontsize=7)
            ax.tick_params(labelsize=6.5, colors="#52514e")
            for s in ["top", "right"]:
                ax.spines[s].set_visible(False)
            ax.grid(axis="y", color="#e4e3df", lw=0.6)
        axes[0][c].set_title(MODELS[m], fontsize=8)
        axes[1][c].set_xticks(xs, [lab.get(v, "+" + v[3:]) for v in vs], fontsize=6.5, rotation=30)
    axes[0][0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(out / f"fig_mixed_precision.{ext}", dpi=300)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--run", required=True)
    b.add_argument("--split", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--rank", required=True, help="17_layer_quant_error.py output dir")
    b.add_argument("--levels", nargs="+", default=DEFAULT_LEVELS)
    b.add_argument("--force", action="store_true")
    l = sp.add_parser("latency")
    l.add_argument("--runs", nargs="+", required=True)
    l.add_argument("--out", required=True)
    l.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    l.add_argument("--reps", type=int, default=200)
    s = sp.add_parser("summarize")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--dumps", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--levels", nargs="+", default=DEFAULT_LEVELS)
    f = sp.add_parser("figure")
    f.add_argument("--out", required=True)
    f.add_argument("--levels", nargs="+", default=DEFAULT_LEVELS)
    a = ap.parse_args()
    {"build": build, "latency": latency, "summarize": summarize, "figure": figure}[a.cmd](a)


if __name__ == "__main__":
    main()
