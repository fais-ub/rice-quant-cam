"""Final paper tables/figures in one folder: results/paper_final (numbered tabNN_ / figNN_)."""
import importlib.util
import os
import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

D = Path(os.environ.get("JOIV_DATA", Path.home() / "joiv-data")).expanduser()
PH = D / "phone"
R = D / "results"
OUT = R / "paper_final"
DATA = OUT / "data"
SCRIPTS = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location("t11", SCRIPTS / "11_phone_tables.py")
t11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t11)

REDMI_V03 = [f"app03/redmi_note11pro5g/bench_Xiaomi_2201116SG_{t}" for t in (1795617069, 1795617774, 1795618155, 1795618476)]
XIAOMI_V03 = [f"app03/xiaomi13t/bench_Xiaomi_2306EPN60G_{t}" for t in (1795610590, 1795611067, 1795611355, 1795611610)]
FINAL = {
    "OPPO Reno12 F 4G": ["oppo_cph2687/bench_*"],
    "Redmi Note 11 Pro 5G": ["redmi_note11pro5g/bench_*"],
    "Moto g45 5G": ["app03/motorola_g45/bench_*"],
    "Xiaomi 13T": XIAOMI_V03,
}
MP = "int8_qdq_mp"
MP_MODELS = ["mnv3l_s0", "effb0_s0"]
PREC = dict(t11.PREC, **{MP: "INT8 QDQ mixed (top-4 FP32)"})
REDMI = "Redmi Note 11 Pro 5G"
OPPO = "OPPO Reno12 F 4G"

lat = pd.concat([t11.load(PH, pats)[0] for pats in FINAL.values()], ignore_index=True)
fid = t11.load(PH, ["oppo_cph2687/bench_*", "app03/motorola_g45/bench_*", *REDMI_V03, *XIAOMI_V03])[1]
redmi3 = t11.load(PH, REDMI_V03)[0]
assert set(lat.phone) == set(FINAL), set(lat.phone)
core = lat[lat.precision != MP]
phones = [v[0] for v in t11.PHONES.values()]

best = core.loc[core.groupby(["phone", "model", "precision"]).median_ms.idxmin()].copy()
best["cfg"] = best.apply(t11.cfg, axis=1)
best["source"] = best.phone.map(lambda p: "app02" if p in (OPPO, REDMI) else "app03")
best["estimated"] = False
mpl = lat[lat.precision == MP]
mpb = mpl.loc[mpl.groupby(["phone", "model"]).median_ms.idxmin()].copy()
mpb["cfg"] = mpb.apply(t11.cfg, axis=1)
mpb["source"] = "app03"
mpb["estimated"] = False
r3 = redmi3[redmi3.precision.isin(["int8_qdq_pct", MP]) & redmi3.model.isin(MP_MODELS)]
r3b = r3.loc[r3.groupby(["model", "precision"]).median_ms.idxmin()]
est = []
for m in MP_MODELS:
    a = r3b[(r3b.model == m) & (r3b.precision == MP)].iloc[0]
    p3 = r3b[(r3b.model == m) & (r3b.precision == "int8_qdq_pct")].iloc[0]
    p2 = best[(best.phone == REDMI) & (best.model == m) & (best.precision == "int8_qdq_pct")].iloc[0]
    row = a.copy()
    row["phone"] = REDMI
    row["median_v03"] = a.median_ms
    row["pct_v03"] = p3.median_ms
    row["pct_v02"] = p2.median_ms
    row["ratio_mp_pct_v03"] = a.median_ms / p3.median_ms
    row["median_ms"] = p2.median_ms * a.median_ms / p3.median_ms
    row["cfg"] = t11.cfg(a)
    row["source"] = "app02 pct x app03 ratio"
    row["estimated"] = True
    est.append(row)
est = pd.DataFrame(est)
allbest = pd.concat([best, mpb, est], ignore_index=True)
keep = ["phone", "model", "precision", "ep", "threads", "cfg", "median_ms", "p90_ms", "model_mb", "source", "estimated",
        "median_v03", "pct_v03", "pct_v02", "ratio_mp_pct_v03"]
allbest[keep].sort_values(["phone", "model", "precision"]).to_csv(DATA / "latency_best_final.csv", index=False)
lat.drop(columns=["device"]).to_csv(DATA / "phone_latency_long_final.csv", index=False)
fid.to_csv(DATA / "phone_fidelity_long_final.csv", index=False)
B = allbest.set_index(["phone", "model", "precision"])
size = lat.groupby(["model", "precision"]).model_mb.first()


def tab_latency():
    cols = "ll r " + " ".join(["rr"] * len(phones))
    out = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
           r"\caption{On-device latency (median of 200 runs, ms) of the best configuration per cell. "
           r"Superscript: execution provider (C = CPU, X = XNNPACK) and threads. "
           r"$\times$: speed-up over FP32 on the same phone. FP16 runs on the CPU EP only (XNNPACK rejects its FP16 convolutions). "
           r"INT8 QDQ mixed: Percentile with the 4 most sensitive activation tensors kept in FP32 (Table~\ref{tab:mixedprec}). "
           r"OPPO Reno12 F 4G: app 0.2 only, no mixed-precision run (--). "
           r"$^\dagger$Redmi Note 11 Pro 5G, mixed: its INT8 QDQ (percentile) value times the mixed/percentile ratio "
           r"measured in a later session (app 0.3), in which this phone ran about 15\,\% slower overall.}",
           r"\label{tab:latency}", r"\begin{tabular}{" + cols + "}", r"\toprule",
           r"Model & Precision & Size (MB) & " + " & ".join([rf"\multicolumn{{2}}{{c}}{{{p}}}" for p in phones]) + r" \\",
           " ".join([rf"\cmidrule(lr){{{4 + 2 * i}-{5 + 2 * i}}}" for i in range(len(phones))]),
           r" & & & " + " & ".join([r"ms & $\times$"] * len(phones)) + r" \\", r"\midrule"]
    for m in t11.MODELS:
        precs = t11.PREC_MAIN + ([MP] if m in MP_MODELS else [])
        for j, p in enumerate(precs):
            cells = []
            for ph in phones:
                if (ph, m, p) not in B.index:
                    cells.append("-- & --")
                    continue
                r = B.loc[(ph, m, p)]
                f32 = B.loc[(ph, m, "fp32")].median_ms
                dag = r"$^\dagger$" if r.estimated else ""
                cells.append(rf"{r.median_ms:.1f}\textsuperscript{{{r.cfg}}}{dag} & {f32 / r.median_ms:.2f}")
            name = rf"\multirow{{{len(precs)}}}{{*}}{{{t11.MODELS[m]}}}" if j == 0 else ""
            out.append(f"{name} & {PREC[p]} & {size[(m, p)]:.2f} & " + " & ".join(cells) + r" \\")
        out.append(r"\midrule" if m != list(t11.MODELS)[-1] else r"\bottomrule")
    out += [r"\end{tabular}", r"\end{table*}"]
    return "\n".join(out) + "\n"


(OUT / "tab06_latency.tex").write_text(tab_latency())


def tab_fidelity():
    t, spread = t11.tab_fidelity(fid[fid.precision != MP])
    f = fid[fid.precision == MP].copy()
    f["phone"] = f.device.map(lambda d: t11.PHONES[d][0])
    n_ph = f.phone.nunique()
    per_ep = f.groupby(["model", "ep"])[["pred_agreement", "cam_spearman_mean", "cam_maxabs_diff"]].agg(lambda s: s.max() - s.min())
    sp_mp = per_ep.max().max()

    def rng(s, fmt, scale=1):
        a, b = s.min() * scale, s.max() * scale
        return f"{a:{fmt}}" if f"{a:{fmt}}" == f"{b:{fmt}}" else f"{a:{fmt}}--{b:{fmt}}"

    lines = t.split("\n")
    res = []
    for i, ln in enumerate(lines):
        for m in MP_MODELS:
            if ln.startswith(rf"\multirow{{6}}{{*}}{{{t11.MODELS[m]}}}"):
                ln = ln.replace(r"\multirow{6}", r"\multirow{7}")
        res.append(ln)
        if "INT8 QDQ (percentile)" in ln:
            blk = next(x for x in reversed(res) if x.startswith(r"\multirow"))
            m = next((k for k in MP_MODELS if t11.MODELS[k] in blk), None)
            if m:
                g = f[f.model == m]
                res.append(f" & {PREC[MP]} & {rng(g.pred_agreement, '.0f', 100)} & {rng(g.cam_spearman_mean, '.3f')} & "
                           f"{rng(g.cam_maxabs_diff, '.3f')}" + r"$^\ddagger$ \\")
    t = "\n".join(res)
    note = (rf" $^\ddagger$INT8 QDQ mixed: measured on {['', 'one', 'two', 'three', 'four'][n_ph]} phones (not on the OPPO); "
            rf"identical across phones (max.\ spread {sp_mp:.0e}), the range is CPU vs.\ XNNPACK EP.")
    t = t.replace(r"}" + "\n" + r"\label{tab:fidelity}", note + "}\n" + r"\label{tab:fidelity}")
    return t, spread, sp_mp


t, spread, sp_mp = tab_fidelity()
(OUT / "tab07_fidelity.tex").write_text(t)
(OUT / "tab05_devices.tex").write_text(t11.tab_devices(pd.concat([lat, redmi3[redmi3.precision.isin(["int8_qdq_pct", MP])]]), phones))

tmp = OUT / "_tmp_fig"
tmp.mkdir(exist_ok=True)
t11.fig_latency(best.reset_index(drop=True), phones, tmp)
t11.fig_threads(core, phones, tmp)
for src, dst in [("fig_latency", "fig02_latency"), ("fig_threads", "fig03_threads")]:
    for ext in ["pdf", "png"]:
        shutil.move(tmp / f"{src}.{ext}", OUT / f"{dst}.{ext}")
tmp.rmdir()

ABL = ["mnv2_s0", "mnv3l_s0", "effb0_s0"]
MAIN = {"s8_minmax_rr": "INT8 QDQ (MinMax)", "s8_minmax": "INT8 QDQ (full-range)", "s8_percentile": "INT8 QDQ (percentile)"}
ab = {r: pd.read_csv(R / f"qdq_ablation_{r}.csv").set_index("variant") for r in ABL}
lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3.5pt}",
         r"\caption{Static INT8 QDQ settings, Paddy Doctor test set, seed 0 (PC, ONNX Runtime CPU). Activations U8 (asymmetric) or "
         r"S8 (symmetric), weights always per-channel symmetric INT8; 7-bit = \texttt{reduce\_range} (7-bit weights). "
         r"Flip = top-1 differs from FP32. Bold settings are the ones carried into the main tables.}",
         r"\label{tab:qdqablation}", r"\begin{tabular}{lll " + " ".join(["rrr"] * len(ABL)) + "}", r"\toprule",
         r"Act. & Calibration & Weights & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{t11.MODELS[r]}}}" for r in ABL) + r" \\",
         " ".join(rf"\cmidrule(lr){{{4 + 3 * i}-{6 + 3 * i}}}" for i in range(len(ABL))),
         r" & & & " + " & ".join([r"Acc & F1 & Flip"] * len(ABL)) + r" \\", r"\midrule"]


def abl_cells(v):
    c = []
    for r in ABL:
        x = ab[r].loc[v]
        c += [f"{100 * x.acc:.1f}", f"{100 * x.macro_f1:.1f}", "--" if v == "fp32" else f"{100 * x.flip_vs_fp32:.1f}"]
    return c


lines.append(r"\multicolumn{3}{l}{FP32} & " + " & ".join(abl_cells("fp32")) + r" \\")
lines.append(r"\midrule")
for act in ["u8", "s8"]:
    for cal in ["minmax", "percentile", "entropy"]:
        for rr in [False, True]:
            v = f"{act}_{cal}{'_rr' if rr else ''}"
            lab = [act.upper(), {"minmax": "MinMax", "percentile": "Percentile", "entropy": "Entropy"}[cal], "7-bit" if rr else "8-bit"]
            if v in MAIN:
                lab = [rf"\textbf{{{x}}}" for x in lab]
            lines.append(" & ".join(lab) + " & " + " & ".join(abl_cells(v)) + r" \\")
    if act == "u8":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
(OUT / "tab02_qdq_ablation.tex").write_text("\n".join(lines) + "\n")

acc = pd.read_csv(R / "paper_pc" / "tab_accuracy.csv").set_index(["model", "precision"])
mps = pd.read_csv(R / "paper_pc" / "mixed_precision" / "mp_summary.csv")
mps = mps[(mps.dataset == "paddy_test") & (mps.variant == "mp_top4")].set_index("model")
pts = allbest[allbest.precision.isin(["fp32", "fp16", "int8_dyn", "int8_qdq", "int8_qdq_fr", "int8_qdq_pct", MP])].copy()
pts["model_key"] = pts.model.str.split("_s").str[0]


def acc_of(mk, p, col):
    if p == MP:
        return 100 * mps.loc[mk, "acc10_mean" if col == "acc_mean" else "acc10_sd"]
    return 100 * acc.loc[(mk, p), col]


pts["acc"] = [acc_of(m, p, "acc_mean") for m, p in zip(pts.model_key, pts.precision)]
pts["acc_sd"] = [acc_of(m, p, "acc_sd") for m, p in zip(pts.model_key, pts.precision)]
pts[["phone", "model", "precision", "cfg", "median_ms", "model_mb", "acc", "acc_sd", "source", "estimated"]] \
    .sort_values(["phone", "model", "precision"]).to_csv(DATA / "tradeoff_points_final.csv", index=False)

MCOL = {"mnv2": "#2a78d6", "mnv3l": "#eb6834", "effl0": "#1baf7a", "effb0": "#eda100"}
MARK = {"fp32": "o", "fp16": "s", "int8_qdq": "X", "int8_qdq_pct": "D", MP: "*"}
SHORT = {"fp32": "FP32", "fp16": "FP16", "int8_qdq": "INT8-QDQ", "int8_qdq_pct": "INT8-QDQ-Pct", MP: "INT8-QDQ-Mixed"}
MSIZE = {MP: 2.2}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
YMIN = 88.0
shown = pts[pts.precision.isin(MARK)]
order = [REDMI, "Moto g45 5G", OPPO, "Xiaomi 13T"]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), sharey=True)
for ax, phone in zip(axes.ravel(), order):
    d = shown[shown.phone == phone]
    dd = d.sort_values("median_ms")
    front, top = [], -1
    for _, r in dd.iterrows():
        if r.acc > top:
            front.append(r)
            top = r.acc
    fr = pd.DataFrame(front)
    ax.step(fr.median_ms, fr.acc.clip(lower=YMIN), where="post", color="#9a9994", lw=0.9, ls=":", zorder=1)
    for _, r in d.iterrows():
        y = max(r.acc, YMIN + 0.4)
        s = 14 * r.model_mb * MSIZE.get(r.precision, 1)
        kw = dict(facecolor="white", edgecolor=MCOL[r.model_key], linewidth=1.3) if r.estimated else \
            dict(color=MCOL[r.model_key], edgecolor="white", linewidth=0.8)
        ax.scatter(r.median_ms, y, s=s, marker=MARK[r.precision], alpha=0.95, zorder=4 if r.precision == MP else 3, **kw)
        if r.acc < YMIN:
            ax.annotate(f"{r.acc:.0f}%", (r.median_ms, y), xytext=(0, -9), textcoords="offset points", ha="center",
                        fontsize=6, color=INK2)
    for mk in ["effb0", "mnv3l"]:
        a = d[(d.model_key == mk) & (d.precision == "int8_qdq_pct")]
        b = d[(d.model_key == mk) & (d.precision == MP)]
        if len(a) and len(b):
            ax.annotate("", xy=(b.median_ms.iloc[0], b.acc.iloc[0]), xytext=(a.median_ms.iloc[0], a.acc.iloc[0]),
                        arrowprops=dict(arrowstyle="->", color=MCOL[mk], lw=0.9, shrinkA=4, shrinkB=6), zorder=2)
    ax.set_xscale("log")
    ax.set_title(phone + (" (no mixed-precision run)" if phone == OPPO else ""), fontsize=8.5, color=INK)
    ax.set_ylim(YMIN - 1.5, 98.5)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}" if str(int(v))[0] in "25" else ""))
    t11.style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.tick_params(which="both", labelsize=7.5)
for ax in axes[1]:
    ax.set_xlabel("Best median latency on the phone (ms, log scale)", fontsize=8, color=INK2)
for ax in axes[:, 0]:
    ax.set_ylabel("PC test accuracy (%, mean of 3 seeds)", fontsize=8, color=INK2)
hm = [plt.Line2D([], [], ls="", marker="o", ms=7, color=MCOL[m], label=t11.MODELS[f"{m}_s0"]) for m in MCOL]
hp = [plt.Line2D([], [], ls="", marker=MARK[p], ms=6 if p != MP else 9, color="#7a7974", label=SHORT[p]) for p in MARK]
hs = [plt.scatter([], [], s=14 * v, color="#bdbcb8", label=f"{v} MB") for v in (4, 9, 16)]
fig.legend(handles=hm, loc="upper center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0), handletextpad=0.3)
fig.legend(handles=hp + hs, loc="upper center", ncol=8, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, 0.965),
           handletextpad=0.3, columnspacing=1.0)
fig.tight_layout(rect=(0, 0.02, 1, 0.9))
fig.text(0.99, 0.005, f"Points below {YMIN:.0f}% sit at the axis floor, labelled with their accuracy. Hollow star: Redmi mixed "
         "precision, scaled from a later session (see the latency table). Arrows: INT8-QDQ-Pct to Mixed.", ha="right", fontsize=5.8, color=INK2)
for ext in ["pdf", "png"]:
    fig.savefig(OUT / f"fig11_tradeoff.{ext}", dpi=300)
plt.close(fig)

P = R / "paper_pc"
COPY = {
    "tab01_accuracy.tex": P / "tab_accuracy.tex",
    "tab03_xai.tex": P / "tab_xai.tex",
    "tab04_insertion_check.tex": P / "tab_insertion_check.tex",
    "fig01_insertion_check": P / "fig_insertion_check",
    "tab08_cross_dataset.tex": P / "cross_dataset" / "tab_cross_dataset.tex",
    "tab09_dedup.tex": P / "cross_dataset" / "duplicates" / "tab_dedup.tex",
    "fig04_cross_dataset": P / "cross_dataset" / "fig_cross_dataset",
    "fig05_mixed_precision": P / "mixed_precision" / "fig_mixed_precision",
    "tab11_layer_sensitivity.tex": P / "mixed_precision" / "layer_error" / "tab_layer_sensitivity.tex",
    "fig06_layer_error": P / "mixed_precision" / "fig_layer_error",
    "tab12_ood.tex": P / "ood_guard" / "tab_ood.tex",
    "tab13_confident.tex": P / "ood_guard" / "tab_confident.tex",
    "fig07_ood": P / "ood_guard" / "fig_ood",
    "tab14_per_class.tex": P / "per_class" / "tab_per_class.tex",
    "fig08_per_class": P / "per_class" / "fig_per_class",
    "tab15_calib_size.tex": OUT / "_regen" / "calib_size" / "tab_calib_size.tex",
    "fig09_calib_size": OUT / "_regen" / "calib_size" / "fig_calib_size",
    "fig10_cams_qualitative": P / "fig_cams_qualitative",
}
for dst, src in COPY.items():
    if dst.endswith(".tex"):
        shutil.copy2(src, OUT / dst)
    else:
        for ext in ["pdf", "png"]:
            shutil.copy2(src.with_suffix(f".{ext}"), OUT / f"{dst}.{ext}")

pcl = pd.read_csv(P / "mixed_precision" / "pc_latency.csv")
assert pcl.run.str.endswith("_s0").all(), pcl.run.unique()
t = (P / "mixed_precision" / "tab_mixed_precision.tex").read_text()
n_nan = t.count(r"{\scriptsize$\pm$nan}")
t = t.replace(r"{\scriptsize$\pm$nan}", "")
t = t.replace("PC latency: onnxruntime CPU, 1 thread, median.", "PC latency: onnxruntime CPU, 1 thread, median, seed 0 only.")
(OUT / "tab10_mixed_precision.tex").write_text(t)

print("fidelity spread core", f"{spread:.1e}", "mp", f"{sp_mp:.1e}", "| nan removed in tab10:", n_nan)
print(est[["model", "median_v03", "pct_v03", "ratio_mp_pct_v03", "pct_v02", "median_ms", "cfg"]].round(3).to_string())
print(sorted(p.name for p in OUT.iterdir()))
