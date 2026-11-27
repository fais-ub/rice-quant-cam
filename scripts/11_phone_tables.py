"""Paper tables (LaTeX) and figures for the on-device benchmark across phones."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PHONES = {
    "OPPO_CPH2687": ("OPPO Reno12 F 4G", "Snapdragon 685", "entry"),
    "Xiaomi_2201116SG": ("Redmi Note 11 Pro 5G", "Snapdragon 695", "mid-range"),
    "motorola_motog455G": ("Moto g45 5G", "Snapdragon 6s Gen 3", "mid-range"),
    "Xiaomi_2306EPN60G": ("Xiaomi 13T", "Dimensity 8200-Ultra", "upper (reference)"),
}
COLORS = {"Redmi Note 11 Pro 5G": "#2a78d6", "Moto g45 5G": "#eb6834", "Xiaomi 13T": "#1baf7a",
          "OPPO Reno12 F 4G": "#eda100"}
MODELS = {"mnv2_s0": "MobileNetV2", "mnv3l_s0": "MobileNetV3-L", "effl0_s0": "EfficientNet-Lite0", "effb0_s0": "EfficientNet-B0"}
PREC = {"fp32": "FP32", "fp16": "FP16", "int8_dyn": "INT8 dynamic", "int8_qdq": "INT8 QDQ (minmax)",
        "int8_qdq_fr": "INT8 QDQ (full-range)", "int8_qdq_pct": "INT8 QDQ (percentile)"}
PREC_MAIN = ["fp32", "fp16", "int8_dyn", "int8_qdq_pct"]


def load(root, patterns):
    dirs = []
    for p in patterns:
        dirs += sorted(Path(root).glob(p))
    L, F = [], []
    for d in dirs:
        L.append(pd.read_csv(d / "latency.csv"))
        if (d / "fidelity_vs_pc.csv").exists():
            F.append(pd.read_csv(d / "fidelity_vs_pc.csv"))
    lat = pd.concat(L)
    lat = lat[lat.load_ms.astype(str) != "ERROR"].copy()
    for c in ["load_ms", "median_ms", "p90_ms", "pss_kb", "model_mb", "temp_start_c", "temp_end_c", "thermal_status"]:
        lat[c] = pd.to_numeric(lat[c])
    lat["phone"] = lat.device.map(lambda d: PHONES[d][0])
    return lat.reset_index(drop=True), pd.concat(F).reset_index(drop=True)


def cfg(r):
    return ("X" if r.ep == "XNNPACK" else "C") + str(int(r.threads))


def tab_latency(lat, phones):
    best = lat.loc[lat.groupby(["phone", "model", "precision"]).median_ms.idxmin()].set_index(["phone", "model", "precision"])
    size = lat.groupby(["model", "precision"]).model_mb.first()
    fp32 = best.median_ms.xs("fp32", level="precision")
    cols = "ll r " + " ".join(["rr"] * len(phones))
    out = [r"\begin{table*}[t]", r"\centering", r"\small",
           r"\caption{On-device latency (median of 200 runs, ms) of the best configuration per cell. "
           r"Superscript: execution provider (C = CPU, X = XNNPACK) and threads. "
           r"$\times$: speed-up over FP32 on the same phone. FP16 runs on the CPU EP only (XNNPACK rejects its FP16 convolutions).}",
           r"\label{tab:latency}", r"\begin{tabular}{" + cols + "}", r"\toprule",
           r"Model & Precision & Size (MB) & " + " & ".join([rf"\multicolumn{{2}}{{c}}{{{p}}}" for p in phones]) + r" \\",
           " ".join([rf"\cmidrule(lr){{{4 + 2 * i}-{5 + 2 * i}}}" for i in range(len(phones))]),
           r" & & & " + " & ".join([r"ms & $\times$"] * len(phones)) + r" \\", r"\midrule"]
    for m in MODELS:
        for j, p in enumerate(PREC_MAIN):
            cells = []
            for ph in phones:
                r = best.loc[(ph, m, p)]
                cells.append(rf"{r.median_ms:.1f}\textsuperscript{{{cfg(r)}}} & {fp32[(ph, m)] / r.median_ms:.2f}")
            name = rf"\multirow{{{len(PREC_MAIN)}}}{{*}}{{{MODELS[m]}}}" if j == 0 else ""
            out.append(f"{name} & {PREC[p]} & {size[(m, p)]:.2f} & " + " & ".join(cells) + r" \\")
        out.append(r"\midrule" if m != list(MODELS)[-1] else r"\bottomrule")
    out += [r"\end{tabular}", r"\end{table*}"]
    return "\n".join(out) + "\n", best.reset_index()


def tab_fidelity(fid):
    k = ["model", "precision"]
    m = ["pred_agreement", "cam_spearman_mean", "cam_maxabs_diff"]
    fid = fid.copy()
    fid["phone"] = fid.device.map(lambda d: PHONES[d][0])
    nph = fid.phone.nunique()
    fid["key"] = fid[m].round(4).astype(str).agg("|".join, axis=1)
    major = fid.groupby(k).key.agg(lambda s: s.value_counts().index[0])
    fid["dev"] = fid.key != fid.set_index(k).index.map(major)
    main = fid[~fid.dev]
    spread = main.groupby(k)[m].agg(lambda s: s.max() - s.min()).max().max()
    g = main.groupby(k)[m].first()
    note = ""
    for (ph, p), d in fid[fid.dev].groupby(["phone", "precision"]):
        a0, a1 = 100 * d.pred_agreement.min(), 100 * d.pred_agreement.max()
        agree = f"{a0:.0f}" if round(a0) == round(a1) else f"{a0:.0f}--{a1:.0f}"
        note += (rf" Exception: {ph}, {PREC[p]} (all models) matches the PC more closely "
                 rf"(agree {agree}\%, "
                 rf"$\rho \ge {d.cam_spearman_mean.min():.4f}$, max $|\Delta| \le {d.cam_maxabs_diff.max():.3f}$): "
                 r"its Cortex-A73/A53 cores lack FP16 arithmetic, so ONNX Runtime computes FP16 in FP32 as on the PC."
                 if d.model.nunique() == len(MODELS) else rf" Exception: {ph}, {PREC[p]}, {', '.join(d.model)}.")
    n = int(fid.n.iloc[0])
    out = [r"\begin{table}[t]", r"\centering", r"\small",
           rf"\caption{{Phone vs.\ PC (ONNX Runtime CPU, x86) on the same ONNX files, {n} test images: top-1 agreement, "
           rf"mean Spearman $\rho$ of the upsampled CAMs, and max.\ absolute CAM difference. Values are identical on all {['', 'one', 'two', 'three', 'four', 'five', 'six'][nph]} phones "
           rf"and on both EPs (max.\ spread {spread:.0e}).{note}}}",
           r"\label{tab:fidelity}", r"\begin{tabular}{llrrr}", r"\toprule",
           r"Model & Precision & Agree (\%) & CAM $\rho$ & max $|\Delta|$ \\", r"\midrule"]
    for m in MODELS:
        for j, p in enumerate(PREC):
            r = g.loc[(m, p)]
            name = rf"\multirow{{{len(PREC)}}}{{*}}{{{MODELS[m]}}}" if j == 0 else ""
            out.append(f"{name} & {PREC[p]} & {100 * r.pred_agreement:.0f} & {r.cam_spearman_mean:.3f} & {r.cam_maxabs_diff:.3f}" + r" \\")
        out.append(r"\midrule" if m != list(MODELS)[-1] else r"\bottomrule")
    out += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n", spread


def tab_devices(lat, phones):
    out = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\caption{Test phones and measurement conditions (airplane mode, charging, 120\,s cool-down between models). "
           r"Temperature = battery sensor; thermal status = Android PowerManager (0 = none).}",
           r"\label{tab:devices}", r"\begin{tabular}{llrrrr}", r"\toprule",
           r"Phone & SoC & $T$ range ($^\circ$C) & max $\Delta T$ & Thermal & PSS (MB) \\", r"\midrule"]
    for ph in phones:
        d = lat[lat.phone == ph]
        soc = [v[1] for v in PHONES.values() if v[0] == ph][0]
        dt = (d.temp_end_c - d.temp_start_c).max()
        out.append(f"{ph} & {soc} & {d.temp_start_c.min():.1f}--{d.temp_end_c.max():.1f} & {dt:.1f} & "
                   f"{int(d.thermal_status.max())} & {d.pss_kb.min() / 1024:.0f}--{d.pss_kb.max() / 1024:.0f}" + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#9a9994")
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)


def fig_latency(best, phones, out):
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.9), sharey=False)
    w = 0.8 / len(phones)
    for ax, m in zip(axes, MODELS):
        for i, ph in enumerate(phones):
            v = [best[(best.phone == ph) & (best.model == m) & (best.precision == p)].median_ms.iloc[0] for p in PREC_MAIN]
            x = [k + (i - (len(phones) - 1) / 2) * w for k in range(len(PREC_MAIN))]
            ax.bar(x, v, w - 0.03, color=COLORS[ph], label=ph, edgecolor="white", linewidth=0.6)
        ax.set_xticks(range(len(PREC_MAIN)), ["FP32", "FP16", "INT8\ndyn", "INT8\nQDQ-pct"])
        ax.set_title(MODELS[m], fontsize=9, color="#0b0b0b")
        style(ax)
    axes[0].set_ylabel("Median latency (ms)", fontsize=8, color="#52514e")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(phones), frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ["pdf", "png"]:
        fig.savefig(out / f"fig_latency.{ext}", dpi=300)
    plt.close(fig)


def fig_threads(lat, phones, out):
    fig, axes = plt.subplots(2, 4, figsize=(10, 4.6), sharex=True)
    for r, p in enumerate(["int8_qdq_pct", "fp32"]):
        for ax, m in zip(axes[r], MODELS):
            for i, ph in enumerate(phones):
                for ep, ls in [("CPU", "-"), ("XNNPACK", "--")]:
                    d = lat[(lat.phone == ph) & (lat.model == m) & (lat.precision == p) & (lat.ep == ep)].sort_values("threads")
                    ax.plot(d.threads, d.median_ms, ls, color=COLORS[ph], linewidth=1.6, marker="o", markersize=3.5,
                            label=f"{ph}, {ep}")
            ax.set_xticks([1, 2, 4])
            ax.set_title(f"{MODELS[m]} - {PREC[p].split(' (')[0] if p == 'fp32' else 'INT8 QDQ-pct'}", fontsize=8.5)
            style(ax)
            if r == 1:
                ax.set_xlabel("Threads", fontsize=8, color="#52514e")
        axes[r][0].set_ylabel("Median latency (ms)", fontsize=8, color="#52514e")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(phones), frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ["pdf", "png"]:
        fig.savefig(out / f"fig_threads.{ext}", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phone-root", required=True)
    ap.add_argument("--dirs", nargs="+", required=True, help="bench folders (globs) relative to --phone-root")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    lat, fid = load(Path(a.phone_root).expanduser(), a.dirs)
    phones = [v[0] for v in PHONES.values() if v[0] in set(lat.phone)]
    t, best = tab_latency(lat, phones)
    (out / "tab_latency.tex").write_text(t)
    t, spread = tab_fidelity(fid)
    (out / "tab_fidelity.tex").write_text(t)
    (out / "tab_devices.tex").write_text(tab_devices(lat, phones))
    fig_latency(best, phones, out)
    fig_threads(lat, phones, out)
    lat.drop(columns=["device"]).to_csv(out / "phone_latency_long.csv", index=False)
    print(f"phones: {phones}; rows: {len(lat)}; fidelity spread across phones/EPs: {spread:.1e}")
    print("written:", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
