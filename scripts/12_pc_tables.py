"""Paper tables and figures for the PC (x86 ONNX Runtime) results, plus the accuracy/latency trade-off."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MODELS = {"mnv2": "MobileNetV2", "mnv3l": "MobileNetV3-L", "effl0": "EfficientNet-Lite0", "effb0": "EfficientNet-B0"}
PREC = {"fp32": "FP32", "fp16": "FP16", "int8_dyn": "INT8 dynamic", "int8_qdq": "INT8 QDQ (MinMax)",
        "int8_qdq_fr": "INT8 QDQ (full-range)", "int8_qdq_pct": "INT8 QDQ (percentile)"}
SHORT = {"fp32": "FP32", "fp16": "FP16", "int8_dyn": "INT8-dyn", "int8_qdq": "INT8-QDQ", "int8_qdq_fr": "INT8-QDQ-FR",
         "int8_qdq_pct": "INT8-QDQ-Pct"}
MCOL = {"mnv2": "#2a78d6", "mnv3l": "#eb6834", "effl0": "#1baf7a", "effb0": "#eda100"}
MARK = {"fp32": "o", "fp16": "s", "int8_dyn": "v", "int8_qdq": "X", "int8_qdq_fr": "P", "int8_qdq_pct": "D"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def runs_of(data):
    out = []
    for m in MODELS:
        for d in sorted((data / "runs").glob(f"{m}_s*")):
            out.append((m, int(d.name.split("_s")[1]), d))
    return out


def ms(s, d=3, scale=1.0):
    r"""'mean $\pm$ sd' string for a series of per-seed values."""
    s = pd.Series(s).dropna() * scale
    if s.empty:
        return "--"
    if len(s) == 1:
        return f"{s.iloc[0]:.{d}f}"
    return f"{s.mean():.{d}f}{{\\scriptsize$\\pm${s.std():.{d}f}}}"


def agg(df, keys, cols):
    g = df.groupby(keys, sort=False)[cols]
    m, s = g.mean(), g.std()
    out = m.add_suffix("_mean").join(s.add_suffix("_sd"))
    out["n_seeds"] = g.size()
    return out.reset_index()


def block_rows(name, rows):
    out = []
    for j, (p, cells) in enumerate(rows):
        lead = rf"\multirow{{{len(rows)}}}{{*}}{{{name}}}" if j == 0 else ""
        out.append(f"{lead} & {p} & " + " & ".join(cells) + r" \\")
    return out


def tab_accuracy(data, out):
    L = []
    for m, seed, d in runs_of(data):
        t = pd.read_csv(d / "eval" / "test_classification.csv").assign(model=m, seed=seed)
        f = d / "onnx"
        t["size_mb"] = [(f / f"{p}.onnx").stat().st_size / 1e6 for p in t.precision]
        L.append(t)
    df = pd.concat(L, ignore_index=True)
    cols = ["acc", "macro_f1", "flip_rate_vs_fp32", "mcnemar_p_vs_fp32", "size_mb"]
    a = agg(df, ["model", "precision"], cols)
    sig = df.groupby(["model", "precision"], sort=False).mcnemar_p_vs_fp32
    a["mcnemar_n_sig05"] = sig.apply(lambda s: int((s < 0.05).sum())).values
    a["mcnemar_p_median"] = sig.median().values
    a["mcnemar_p_max"] = sig.max().values
    a.to_csv(out / "tab_accuracy.csv", index=False)
    df.to_csv(out / "pc_classification_long.csv", index=False)
    nseed = df.seed.nunique()

    def pfmt(p):
        return "--" if p >= 0.9999 else (f"{p:.2f}" if p >= 0.01 else f"$10^{{{int(np.floor(np.log10(max(p, 1e-300))))}}}$")

    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
             rf"\caption{{Test-set classification on the PC (ONNX Runtime CPU, $n$ = {int(df.n.iloc[0])}), mean $\pm$ sd over "
             rf"{nseed} seeds. Flip = share of images whose top-1 differs from FP32. McNemar vs.\ FP32: number of seeds with "
             r"$p<0.05$ and the median $p$ (exact binomial test on discordant pairs).}",
             r"\label{tab:accuracy}", r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Model & Precision & MB & Acc (\%) & Macro-F1 (\%) & Flip (\%) & \multicolumn{2}{c}{McNemar} \\",
             r"\cmidrule(lr){7-8}", r" & & & & & & sig. & med.\ $p$ \\", r"\midrule"]
    for m in MODELS:
        rows = []
        for p in PREC:
            g = df[(df.model == m) & (df.precision == p)]
            r = a[(a.model == m) & (a.precision == p)].iloc[0]
            rows.append((PREC[p], [f"{r.size_mb_mean:.1f}", ms(g.acc, 1, 100), ms(g.macro_f1, 1, 100),
                                   ms(g.flip_rate_vs_fp32, 1, 100),
                                   "--" if p == "fp32" else f"{r.mcnemar_n_sig05}/{nseed}",
                                   "--" if p == "fp32" else pfmt(r.mcnemar_p_median)]))
        lines += block_rows(MODELS[m], rows)
        lines.append(r"\midrule" if m != list(MODELS)[-1] else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table}"]
    (out / "tab_accuracy.tex").write_text("\n".join(lines) + "\n")
    return df, a


def tab_xai(data, out):
    L = []
    for m, seed, d in runs_of(data):
        L.append(pd.read_csv(d / "eval" / "test_heatmap_summary.csv").assign(model=m, seed=seed))
    df = pd.concat(L, ignore_index=True)
    stab = ["spearman", "ssim", "iou_top10", "iou_top20"]
    faith = ["deletion_auc", "insertion_auc", "deletion_auc_random", "insertion_auc_random"]
    w = df.pivot_table(index=["model", "seed", "precision"], columns="scope", values=stab + faith + ["n"]).reset_index()
    w.columns = [c if not s else f"{c}_{s}" for c, s in w.columns]
    w["same_pred_share"] = w.n_same_pred / w.n_all
    keep = [f"{c}_all" for c in stab] + [f"{c}_same_pred" for c in stab] + [f"{c}_all" for c in faith] + ["same_pred_share"]
    a = agg(w, ["model", "precision"], keep)
    a.to_csv(out / "tab_xai.csv", index=False)
    w.to_csv(out / "pc_xai_long.csv", index=False)
    nseed = w.seed.nunique()
    hdr = r"$\rho$ & SSIM & IoU$_{10}$ & IoU$_{20}$"
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             rf"\caption{{Grad-CAM stability against FP32 on all {int(df.n.max())} test images and on the images whose "
             r"prediction does not change (same pred., share in column \%), and faithfulness on 300 test images: "
             r"deletion AUC (lower is better) and insertion AUC (higher is better), each precision judged by its own "
             r"predictions. Random: a random $7\times7$ map, upsampled, judged by the FP32 model. "
             rf"Mean $\pm$ sd over {nseed} seeds. Insertion AUC is not comparable across precisions as such "
             r"(Table~\ref{tab:inscheck}).}",
             r"\label{tab:xai}", r"\begin{tabular}{ll rrrr r rrrr rr}", r"\toprule",
             r"& & \multicolumn{4}{c}{All images} & & \multicolumn{4}{c}{Same prediction as FP32} & \multicolumn{2}{c}{Faithfulness} \\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){8-11}\cmidrule(lr){12-13}",
             rf"Model & Precision & {hdr} & \% & {hdr} & Del.\,$\downarrow$ & Ins.\,$\uparrow$ \\", r"\midrule"]
    for m in MODELS:
        rows = []
        for p in list(PREC) + ["random"]:
            if p == "random":
                g = w[(w.model == m) & (w.precision == "fp32")]
                rows.append(("Random (FP32)", ["--"] * 9 + [ms(g.deletion_auc_random_all), ms(g.insertion_auc_random_all)]))
                continue
            g = w[(w.model == m) & (w.precision == p)]
            if p == "fp32":
                cells = ["--"] * 9
            else:
                cells = ([ms(g[f"{c}_all"], 2) for c in stab] + [ms(g.same_pred_share, 0, 100)]
                         + [ms(g[f"{c}_same_pred"], 2) for c in stab])
            rows.append((PREC[p], cells + [ms(g.deletion_auc_all), ms(g.insertion_auc_all)]))
        lines += block_rows(MODELS[m], rows)
        lines.append(r"\midrule" if m != list(MODELS)[-1] else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table*}"]
    (out / "tab_xai.tex").write_text("\n".join(lines) + "\n")
    return w, a


def tab_insertion_check(data, out):
    L = []
    for m, seed, d in runs_of(data):
        f = d / "eval" / "insertion_check.csv"
        if f.exists():
            L.append(pd.read_csv(f).assign(model=m, seed=seed))
    if not L:
        print("no insertion_check.csv yet")
        return None, None
    df = pd.concat(L, ignore_index=True)
    common = set.intersection(*[set(g.seed) for _, g in df.groupby("model")])
    if df.model.nunique() < len(MODELS):
        common = set()
    df = df[df.seed.isin(common)] if common else df
    df["ins_gain_rand"] = df.ins_own - df.ins_rand
    df["del_gain_rand"] = df.del_rand - df.del_own
    df["ins_above_blur"] = df.ins_own - df.conf_blur
    df["ins_rel_full"] = df.ins_own / df.conf_full
    span = (df.conf_full - df.conf_blur)
    df["ins_norm"] = np.where(span > 0.05, (df.ins_own - df.conf_blur) / span.where(span > 0.05), np.nan)
    df["ins_rand_norm"] = np.where(span > 0.05, (df.ins_rand - df.conf_blur) / span.where(span > 0.05), np.nan)
    df["ins_gain_norm"] = df.ins_norm - df.ins_rand_norm
    df["ins_map_gain"] = df.ins_own - df.ins_fp32map
    df.to_csv(out / "pc_insertion_check_long.csv", index=False)
    ref = df[df.precision == "fp32"].set_index(["model", "seed", "idx"]).ins_own
    df["ins_fp32own_same"] = df.set_index(["model", "seed", "idx"]).index.map(ref)
    df.loc[df.ins_own_at_fp32.isna(), "ins_fp32own_same"] = np.nan
    df["ins_map_at_fp32_diff"] = df.ins_own_at_fp32 - df.ins_fp32own_same

    cols = ["conf_full", "conf_blur", "maxprob", "entropy", "ins_own", "ins_rand", "ins_gain_rand", "ins_above_blur",
            "ins_rel_full", "ins_norm", "ins_rand_norm", "ins_gain_norm", "ins_fp32map", "ins_map_gain",
            "ins_own_at_fp32", "ins_fp32own_same", "ins_map_at_fp32_diff", "del_own", "del_rand", "del_gain_rand", "correct", "same_pred_as_fp32"]
    per_seed = df.groupby(["model", "precision", "seed"], sort=False)[cols].mean().reset_index()
    a = agg(per_seed, ["model", "precision"], cols)
    a.to_csv(out / "tab_insertion_check.csv", index=False)
    per_seed.to_csv(out / "pc_insertion_check_per_seed.csv", index=False)
    nseed = df.seed.nunique()
    precs = ["fp32", "fp16", "int8_dyn", "int8_qdq", "int8_qdq_pct"]
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             rf"\caption{{Is a higher insertion AUC under INT8 a better heatmap? 300 test images, "
             rf"{'seed 0' if nseed == 1 else f'mean $\\pm$ sd over {nseed} seeds'}. Each precision tracks its own top-1 class $c$. "
             r"$p_\mathrm{full}$, $p_\mathrm{blur}$: softmax of $c$ on the full and on the fully blurred image (end and start of the insertion curve). "
             r"Ins.: insertion AUC of the model's own CAM; Rand.: same model, a random $7\times7$ map (the same map for all precisions); "
             r"$\Delta_\mathrm{rand}$ = Ins.\ $-$ Rand. Norm.: per image $(\mathrm{Ins.}-p_\mathrm{blur})/(p_\mathrm{full}-p_\mathrm{blur})$ "
             r"(images with $p_\mathrm{full}-p_\mathrm{blur}>0.05$), own CAM and random map. "
             r"Swap tests: FP32 map = model $p$ judged with FP32's CAM for class $c$; "
             r"$\Delta_{@\mathrm{FP32}}$ = insertion of $p$'s CAM minus FP32's own CAM, both judged by the FP32 model "
             r"(images where $p$ and FP32 agree). Del.\ gain = Del.$_\mathrm{rand}-$Del.$_\mathrm{own}$.}",
             r"\label{tab:inscheck}", r"\begin{tabular}{ll rr rrr rr rr r}", r"\toprule",
             r"& & \multicolumn{2}{c}{Confidence} & \multicolumn{3}{c}{Insertion AUC} & \multicolumn{2}{c}{Normalised ins.} & \multicolumn{2}{c}{Map swap} & \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
             r"Model & Precision & $p_\mathrm{full}$ & $p_\mathrm{blur}$ & Ins. & Rand. & $\Delta_\mathrm{rand}$ & own & rand. & FP32 map & $\Delta_{@\mathrm{FP32}}$ & Del.\ gain \\",
             r"\midrule"]
    for m in MODELS:
        rows = []
        for p in precs:
            g = per_seed[(per_seed.model == m) & (per_seed.precision == p)]
            if g.empty:
                continue
            rows.append((PREC[p], [ms(g[c]) for c in ["conf_full", "conf_blur", "ins_own", "ins_rand", "ins_gain_rand",
                                                       "ins_norm", "ins_rand_norm", "ins_fp32map"]]
                         + ["--" if p == "fp32" else ms(g.ins_map_at_fp32_diff), ms(g.del_gain_rand)]))
        if rows:
            lines += block_rows(MODELS[m], rows)
            lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}"]
    (out / "tab_insertion_check.tex").write_text("\n".join(lines) + "\n")
    return df, a


def fig_insertion_check(data, out, model="effb0", seed=0):
    f = data / "runs" / f"{model}_s{seed}" / "eval" / "insertion_check_curves.npz"
    if not f.exists():
        return
    z = np.load(f)
    fr = np.linspace(0, 1, z["fp32__own__ins"].shape[1])
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.3), sharey=True)
    for ax, kind, title in [(axes[0], "ins", "Insertion"), (axes[1], "del", "Deletion")]:
        for p, col in [("fp32", MCOL["mnv2"]), ("int8_qdq", MCOL["mnv3l"]), ("int8_qdq_pct", MCOL["effl0"])]:
            ax.plot(fr, z[f"{p}__own__{kind}"].mean(0), "-", color=col, lw=2, marker="o", ms=4, label=f"{SHORT[p]}, own CAM")
            ax.plot(fr, z[f"{p}__rand__{kind}"].mean(0), "--", color=col, lw=1.4, label=f"{SHORT[p]}, random map")
            if p != "fp32":
                ax.plot(fr, z[f"{p}__fp32map__{kind}"].mean(0), ":", color=col, lw=1.6, label=f"{SHORT[p]}, FP32's CAM")
        ax.set_title(f"{title} ({MODELS[model]}, seed {seed})", fontsize=8.5, color=INK)
        ax.set_xlabel("Fraction of pixels " + ("inserted" if kind == "ins" else "deleted"), fontsize=8, color=INK2)
        style(ax)
    axes[0].set_ylabel("Softmax of tracked class", fontsize=8, color=INK2)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    for ext in ["pdf", "png"]:
        fig.savefig(out / f"fig_insertion_check.{ext}", dpi=300)
    plt.close(fig)


def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#9a9994")
    ax.tick_params(which="both", colors=INK2, labelsize=7.5)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_tradeoff(data, out, acc, ymin=88.0, precs=("fp32", "fp16", "int8_qdq", "int8_qdq_pct")):
    ph = pd.read_csv(data / "phone" / "all_phones_best_config.csv")
    ph["model_key"] = ph.model.str.split("_s").str[0]
    acc = acc.set_index(["model", "precision"])
    ph["acc"] = [100 * acc.loc[(m, p), "acc_mean"] for m, p in zip(ph.model_key, ph.precision)]
    ph["acc_sd"] = [100 * acc.loc[(m, p), "acc_sd"] for m, p in zip(ph.model_key, ph.precision)]
    ph.to_csv(out / "tradeoff_points.csv", index=False)
    ph = ph[ph.precision.isin(precs)]
    order = ["Redmi Note 11 Pro 5G", "Moto g45 5G", "OPPO Reno12 F 4G", "Xiaomi 13T"]
    phones = [next(p for p in ph.phone.unique() if p.startswith(o)) for o in order if any(ph.phone.str.startswith(o))]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharey=True)
    for ax, phone in zip(axes.ravel(), phones):
        d = ph[ph.phone == phone]
        dd = d.sort_values("median_ms")
        front, best = [], -1
        for _, r in dd.iterrows():
            if r.acc > best:
                front.append(r)
                best = r.acc
        fr = pd.DataFrame(front)
        ax.step(fr.median_ms, fr.acc.clip(lower=ymin), where="post", color="#9a9994", lw=0.9, ls=":", zorder=1)
        for _, r in d.iterrows():
            y = max(r.acc, ymin + 0.4)
            ax.scatter(r.median_ms, y, s=14 * r.model_mb, marker=MARK[r.precision], color=MCOL[r.model_key],
                       edgecolor="white", linewidth=0.8, alpha=0.9, zorder=3)
            if r.acc < ymin:
                ax.annotate(f"{r.acc:.0f}%", (r.median_ms, y), xytext=(0, -9), textcoords="offset points",
                            ha="center", fontsize=6, color=INK2)
        ax.set_xscale("log")
        ax.set_title(phone, fontsize=8.5, color=INK)
        ax.set_ylim(ymin - 1.5, 98.5)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}" if str(int(v))[0] in "25" else ""))
        style(ax)
    for ax in axes[1]:
        ax.set_xlabel("Best median latency on the phone (ms, log scale)", fontsize=8, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("PC test accuracy (%, mean of 3 seeds)", fontsize=8, color=INK2)
    hm = [plt.Line2D([], [], ls="", marker="o", ms=7, color=MCOL[m], label=MODELS[m]) for m in MODELS]
    hp = [plt.Line2D([], [], ls="", marker=MARK[p], ms=6, color="#7a7974", label=SHORT[p]) for p in precs]
    hs = [plt.scatter([], [], s=14 * v, color="#bdbcb8", label=f"{v} MB") for v in (4, 9, 16)]
    fig.legend(handles=hm, loc="upper center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.3)
    fig.legend(handles=hp + hs, loc="upper center", ncol=7, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.965),
               handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.01, 1, 0.9))
    fig.text(0.99, 0.005, f"Points below {ymin:.0f}% are drawn at the axis floor and labelled with their accuracy.",
             ha="right", fontsize=6, color=INK2)
    for ext in ["pdf", "png"]:
        fig.savefig(out / f"fig_tradeoff.{ext}", dpi=300)
    plt.close(fig)
    return ph


def runvar(data, out, dirs):
    L = []
    for i, d in enumerate(dirs):
        t = pd.read_csv(data / d / "latency.csv")
        t = t[t.load_ms.astype(str) != "ERROR"].copy()
        t["median_ms"] = pd.to_numeric(t.median_ms)
        L.append(t.assign(rep=i, folder=Path(d).name))
    df = pd.concat(L, ignore_index=True)
    g = df.groupby(["model", "precision", "ep", "threads"]).median_ms
    r = g.agg(["count", "mean", "std", "min", "max"]).reset_index()
    r = r[r["count"] > 1]
    r["cv_pct"] = 100 * r["std"] / r["mean"]
    r["range_pct"] = 100 * (r["max"] - r["min"]) / r["mean"]
    best = df.loc[df.groupby(["rep", "model", "precision"]).median_ms.idxmin()]
    best["cfg"] = best.ep.str[0] + best.threads.astype(str)
    bc = best.groupby(["model", "precision"]).agg(best_cfgs=("cfg", lambda s: "/".join(s)),
                                                  best_ms=("median_ms", lambda s: "/".join(f"{v:.1f}" for v in s))).reset_index()
    r.to_csv(out / "xiaomi13t_runvar.csv", index=False)
    bc.to_csv(out / "xiaomi13t_runvar_best.csv", index=False)
    return r, bc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runvar-dirs", nargs="*", default=[])
    a = ap.parse_args()
    data, out = Path(a.data).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)

    _, acc = tab_accuracy(data, out)
    print(acc[["model", "precision", "acc_mean", "acc_sd", "flip_rate_vs_fp32_mean", "mcnemar_n_sig05", "mcnemar_p_median"]].round(4).to_string())
    _, xai = tab_xai(data, out)
    print(xai[["model", "precision", "spearman_all_mean", "iou_top20_all_mean", "spearman_same_pred_mean",
               "deletion_auc_all_mean", "insertion_auc_all_mean", "insertion_auc_random_all_mean"]].round(3).to_string())
    _, ic = tab_insertion_check(data, out)
    if ic is not None:
        print(ic[["model", "precision", "n_seeds", "conf_full_mean", "conf_blur_mean", "ins_own_mean", "ins_rand_mean",
                  "ins_gain_rand_mean", "ins_norm_mean", "ins_rand_norm_mean", "ins_fp32map_mean", "ins_own_at_fp32_mean",
                  "ins_fp32own_same_mean", "del_gain_rand_mean"]].round(3).to_string())
    fig_insertion_check(data, out)
    fig_tradeoff(data, out, acc)
    if a.runvar_dirs:
        r, bc = runvar(data, out, a.runvar_dirs)
        print(r.round(2).to_string())
        print(bc.to_string())
        print(f"run-to-run CV of median latency: median {r.cv_pct.median():.1f}%, max {r.cv_pct.max():.1f}%")
    print("written:", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
