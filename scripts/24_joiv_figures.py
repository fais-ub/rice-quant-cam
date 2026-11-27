"""All paper figures re-rendered in JOIV style -> results/paper_final/joiv/."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager, ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

D = Path(os.environ.get("JOIV_DATA", Path.home() / "joiv-data")).expanduser()
R = D / "results"
P = R / "paper_pc"
F = R / "paper_final"
OUT = F / "joiv"
OUT.mkdir(parents=True, exist_ok=True)

FONT_DIRS = [os.environ.get("FONT_DIR", ""), "C:/Windows/Fonts", "/mnt/c/Windows/Fonts"]
for f in ["times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"]:
    p = next((Path(d) / f for d in FONT_DIRS if d and (Path(d) / f).exists()), None)
    if p:
        font_manager.fontManager.addfont(str(p))
FS = 8
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FS,
    "ytick.labelsize": FS, "legend.fontsize": FS, "legend.title_fontsize": FS, "figure.titlesize": FS,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
    "hatch.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.handlelength": 1.8, "legend.columnspacing": 1.0, "legend.handletextpad": 0.4,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "axes.titlepad": 3,
})
MM = 1 / 25.4
W1, W2 = 86 * MM, 181 * MM
INK, GRID, ECOL = "#000000", "#d9d9d9", "#555555"
SLOT = [("#000000", "D", ""), ("#0072B2", "o", ""), ("#E69F00", "s", ""), ("#B4DDF5", "^", "")]


def tint(c, f=0.55):
    """Solid lighter version of a colour (mix with white)."""
    r, g, b = matplotlib.colors.to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)
MODELS = {"mnv2": "MobileNetV2", "mnv3l": "MobileNetV3-L", "effl0": "EfficientNet-Lite0", "effb0": "EfficientNet-B0"}
MSLOT = {"mnv2": 1, "mnv3l": 2, "effl0": 3, "effb0": 0}
PHONES = ["OPPO Reno12 F 4G", "Redmi Note 11 Pro 5G", "Moto g45 5G", "Xiaomi 13T"]
PSLOT = {"OPPO Reno12 F 4G": 3, "Redmi Note 11 Pro 5G": 1, "Moto g45 5G": 2, "Xiaomi 13T": 0}
MP = "int8_qdq_mp"


def mc(m):
    return SLOT[MSLOT[m]][0]


def mm_(m):
    return SLOT[MSLOT[m]][1]


def mh(m):
    return SLOT[MSLOT[m]][2]


def pc(p):
    return SLOT[PSLOT[p]][0]


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def save(fig, name, exts=("pdf", "png")):
    for ext in exts:
        fig.savefig(OUT / f"{name}.{ext}", dpi=600)
    plt.close(fig)
    print("written", name)


def bar(ax, x, h, w, m=None, color=None, hatch=None, **kw):
    return ax.bar(x, h, w, color=color or mc(m), hatch=mh(m) if hatch is None else hatch, edgecolor=INK, linewidth=0.4,
                  error_kw=dict(lw=0.6, capsize=1.2, ecolor=ECOL), **kw)


def model_handles(kind="bar"):
    if kind == "bar":
        return [plt.Rectangle((0, 0), 1, 1, fc=mc(m), hatch=mh(m), ec=INK, lw=0.4, label=MODELS[m]) for m in MODELS]
    return [plt.Line2D([], [], color=mc(m), marker=mm_(m), ms=3.5, mec=INK, mew=0.3, label=MODELS[m]) for m in MODELS]


CLS_FULL = {"bacterial_leaf_blight": "Bact. leaf blight", "bacterial_leaf_streak": "Bact. leaf streak",
            "bacterial_panicle_blight": "Bact. panicle blight", "blast": "Blast", "brown_spot": "Brown spot",
            "dead_heart": "Dead heart", "downy_mildew": "Downy mildew", "hispa": "Hispa", "normal": "Normal", "tungro": "Tungro"}
CLS_AB = {"bacterial_leaf_blight": "BLB", "bacterial_leaf_streak": "BLS", "bacterial_panicle_blight": "BPB", "blast": "Blast",
          "brown_spot": "BS", "dead_heart": "DH", "downy_mildew": "DM", "hispa": "Hispa", "normal": "Normal", "tungro": "Tungro"}


def fig_cams(interp="none", exts=("pdf",)):
    from PIL import Image
    d = np.load(P / "cams_qualitative.npz")
    runs, kinds, labels, imgs = list(d["runs"]), d["kinds"], d["labels"], d["images"]
    precs = ["fp32", "int8_qdq", "int8_qdq_pct"]
    plab = {"fp32": "FP32", "int8_qdq": "INT8 MinMax", "int8_qdq_pct": "INT8 Pct"}
    klab = {"flip": "MinMax flip", "drift": "CAM drift", "stable": "stable"}
    n, ncol = len(labels), 1 + len(runs) * len(precs)
    left, right, bottom, sp = 0.105, 0.997, 0.004, 0.035
    cw = W2 * (right - left) / (ncol + (ncol - 1) * sp)
    top_in = 0.36
    H = n * cw + (n - 1) * sp * cw + top_in + bottom * 7
    fig, axes = plt.subplots(n, ncol, figsize=(W2, H), squeeze=False)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=1 - top_in / H, wspace=sp, hspace=sp)

    def up(cam):
        m = np.asarray(Image.fromarray(cam.astype(np.float32), mode="F").resize((224, 224), Image.BILINEAR), dtype=np.float32)
        lo, hi = m.min(), m.max()
        return (m - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(m)

    for i in range(n):
        ax = axes[i][0]
        ax.imshow(imgs[i], interpolation=interp)
        ax.set_ylabel(f"{CLS_FULL[labels[i]]}\n({klab[kinds[i]]})", labelpad=2)
        for j, run in enumerate(runs):
            classes = list(d[f"{run}__classes"])
            for k, prec in enumerate(precs):
                ax = axes[i][1 + j * len(precs) + k]
                heat = plt.get_cmap("inferno")(up(d[f"{run}__{prec}__cam"][i]))[..., :3]
                ax.imshow(0.45 * imgs[i].astype(np.float32) / 255 + 0.55 * heat, interpolation=interp)
                c, cf = int(d[f"{run}__{prec}__pred"][i]), float(d[f"{run}__{prec}__conf"][i])
                ok = classes[c] == labels[i]
                ax.text(0.03, 0.03, f"{'' if ok else '× '}{CLS_AB[classes[c]]} {cf:.2f}", transform=ax.transAxes,
                        color="white" if ok else INK, va="bottom", ha="left", fontweight="normal" if ok else "bold",
                        bbox=dict(boxstyle="square,pad=0.15", fc=INK if ok else "white", ec=INK, lw=0.4, alpha=0.85))
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0][0].set_title("Input")
    for j, run in enumerate(runs):
        for k, prec in enumerate(precs):
            axes[0][1 + j * len(precs) + k].set_title(plab[prec])
        l = axes[0][1 + j * len(precs)].get_position()
        r = axes[0][(j + 1) * len(precs)].get_position()
        fig.text((l.x0 + r.x1) / 2, r.y1 + 0.17 / H * 1.0 + 0.012, f"{MODELS[run.split('_')[0]]} (seed {run[-1]})",
                 ha="center", va="bottom", fontweight="bold")
        fig.add_artist(plt.Line2D([l.x0 - sp * cw / W2 / 2] * 2, [bottom, r.y1 + 0.03], color=INK, lw=0.5))
    save(fig, "Fig1_fig10_cams_qualitative", exts)


def fig_insertion():
    z = np.load(D / "runs" / "effb0_s0" / "eval" / "insertion_check_curves.npz")
    fr = np.linspace(0, 1, z["fp32__own__ins"].shape[1])
    fig, axes = plt.subplots(1, 2, figsize=(W1, 2.75), sharey=True, layout="constrained")
    cols = {"fp32": (SLOT[0][0], "o", "FP32"), "int8_qdq": (SLOT[2][0], "s", "INT8 MinMax"), "int8_qdq_pct": (SLOT[1][0], "^", "INT8 Pct")}
    me = max(1, len(fr) // 8)
    for ax, kind, title in [(axes[0], "ins", "Insertion"), (axes[1], "del", "Deletion")]:
        for p, (col, mk, lab) in cols.items():
            ax.plot(fr, z[f"{p}__own__{kind}"].mean(0), "-", color=col, lw=1.1, marker=mk, ms=3, markevery=me, mec=INK, mew=0.3)
            ax.plot(fr, z[f"{p}__rand__{kind}"].mean(0), "--", color=col, lw=0.9)
            if p != "fp32":
                ax.plot(fr, z[f"{p}__fp32map__{kind}"].mean(0), ":", color=col, lw=1.1)
        ax.set_title(title)
        ax.set_xlabel("Fraction of pixels " + ("inserted" if kind == "ins" else "deleted"))
        grid(ax, "both")
    axes[0].set_ylabel("Softmax of tracked class")
    hp = [plt.Line2D([], [], color=c, marker=mk, ms=3, mec=INK, mew=0.3, label=lab) for c, mk, lab in cols.values()]
    hl = [plt.Line2D([], [], color="#808080", ls=ls, lw=1, label=lab) for ls, lab in [("-", "own CAM"), ("--", "random map"),
                                                                               (":", "FP32's CAM")]]
    fig.legend(handles=hp + hl, loc="outside upper center", ncol=3, handlelength=2.2)
    save(fig, "Fig2_fig01_insertion_check")


def fig_layer_error():
    L = pd.concat([pd.read_csv(f) for f in sorted((P / "mixed_precision" / "layer_error").glob("*/layer_error.csv"))],
                  ignore_index=True)
    L["model"] = L.run.str.split("_s").str[0]
    ms = ["effb0", "mnv3l", "mnv2", "effl0"]
    fig, axes = plt.subplots(len(ms), 2, figsize=(W1, 5.4), layout="constrained")
    sty = {"pct": dict(color=SLOT[2][0], lw=1.2, label="Percentile", zorder=2),
           "minmax": dict(color=INK, lw=0.55, label="MinMax", zorder=3)}
    for r, m in enumerate(ms):
        se_x = None
        for tag in ["pct", "minmax"]:
            l = L[(L.model == m) & (L.qmodel == tag)].sort_values("order").reset_index(drop=True)
            xs = np.arange(len(l))
            axes[r][0].plot(xs, l.steps_p999.clip(lower=0.5), **sty[tag])
            axes[r][1].plot(xs, l.cum_sqnr_db.clip(-5, 60), **sty[tag])
            if tag == "minmax":
                se_x = xs[l.role.str.startswith("se").to_numpy()]
        for ax in axes[r]:
            for x in se_x:
                ax.axvspan(x - 0.5, x + 0.5, color="#e6e6e6", lw=0, zorder=0)
            ax.set_xlim(-1, len(xs))
        axes[r][0].set_yscale("log")
        axes[r][0].set_ylim(0.3, 600)
        axes[r][1].set_ylim(-5, 62)
        axes[r][0].set_ylabel(MODELS[m], fontweight="bold")
        axes[r][0].yaxis.set_major_locator(ticker.LogLocator(numticks=4))
    axes[0][0].set_title("INT8 steps covering 99.9 %")
    axes[0][1].set_title("SQNR vs FP32 so far (dB)")
    for ax in axes[-1]:
        ax.set_xlabel("Activation tensor (graph order)")
    h = [plt.Line2D([], [], **{k: v for k, v in sty[t].items() if k != "zorder"}) for t in ["minmax", "pct"]]
    h.append(plt.Rectangle((0, 0), 1, 1, fc="#e6e6e6", lw=0, label="SE block"))
    fig.legend(handles=h, loc="outside upper center", ncol=3)
    save(fig, "Fig3_fig06_layer_error")


lat_all = pd.read_csv(F / "data" / "phone_latency_long_final.csv")
core = lat_all[lat_all.precision != MP]
best_all = pd.read_csv(F / "data" / "latency_best_final.csv")
PMOD = {"mnv2_s0": "mnv2", "mnv3l_s0": "mnv3l", "effl0_s0": "effl0", "effb0_s0": "effb0"}
PREC_MAIN = ["fp32", "fp16", "int8_dyn", "int8_qdq_pct"]


def phone_handles(ep=True):
    h = [plt.Line2D([], [], color=pc(p), marker=SLOT[PSLOT[p]][1], ms=3.5, mec=INK, mew=0.3, label=p) for p in PHONES]
    if ep:
        h += [plt.Line2D([], [], color=INK, ls="-", lw=1, label="CPU EP"), plt.Line2D([], [], color=INK, ls="--", lw=1, label="XNNPACK EP")]
    return h


def fig_threads():
    fig, axes = plt.subplots(4, 2, figsize=(W1, 5.6), sharex=True, layout="constrained")
    for c, p in enumerate(["int8_qdq_pct", "fp32"]):
        for r, m in enumerate(PMOD):
            ax = axes[r][c]
            for ph in PHONES:
                for ep, ls in [("CPU", "-"), ("XNNPACK", "--")]:
                    d = core[(core.phone == ph) & (core.model == m) & (core.precision == p) & (core.ep == ep)].sort_values("threads")
                    ax.plot(d.threads, d.median_ms, ls, color=pc(ph), lw=1.0, marker=SLOT[PSLOT[ph]][1], ms=3, mec=INK, mew=0.3)
            ax.set_xticks([1, 2, 4])
            ax.set_ylim(bottom=0)
            grid(ax)
            if r == 0:
                ax.set_title("INT8 QDQ-Pct" if p != "fp32" else "FP32")
            if c == 0:
                ax.set_ylabel(f"{MODELS[PMOD[m]]}\nlatency (ms)")
    for ax in axes[-1]:
        ax.set_xlabel("Threads")
    fig.legend(handles=phone_handles(), loc="outside upper center", ncol=2)
    save(fig, "Fig4_fig03_threads")


def fig_tradeoff():
    pts = pd.read_csv(F / "data" / "tradeoff_points_final.csv")
    pts["mk"] = pts.model.str.split("_s").str[0]
    MARK = {"fp32": "o", "fp16": "s", "int8_qdq": "X", "int8_qdq_pct": "D", MP: "*"}
    SHORT = {"fp32": "FP32", "fp16": "FP16", "int8_qdq": "INT8-QDQ", "int8_qdq_pct": "INT8-QDQ-Pct", MP: "INT8-QDQ-Mixed"}
    MSIZE = {MP: 2.2}
    YMIN = 88.0
    shown = pts[pts.precision.isin(MARK)]
    order = ["Redmi Note 11 Pro 5G", "Moto g45 5G", "OPPO Reno12 F 4G", "Xiaomi 13T"]
    fig, axes = plt.subplots(2, 2, figsize=(W2, 5.3), sharey=True, layout="constrained")
    for ax, phone in zip(axes.ravel(), order):
        d = shown[shown.phone == phone]
        front, top = [], -1
        for _, r in d.sort_values("median_ms").iterrows():
            if r.acc > top:
                front.append(r)
                top = r.acc
        fr = pd.DataFrame(front)
        ax.step(fr.median_ms, fr.acc.clip(lower=YMIN), where="post", color="#808080", lw=0.8, ls=":", zorder=1)
        for _, r in d.iterrows():
            y = max(r.acc, YMIN + 0.4)
            s = 11 * r.model_mb * MSIZE.get(r.precision, 1)
            kw = dict(facecolor="white", edgecolor=mc(r.mk), linewidth=1.2) if r.estimated else \
                dict(facecolor=mc(r.mk), edgecolor=INK, linewidth=0.4)
            ax.scatter(r.median_ms, y, s=s, marker=MARK[r.precision], alpha=0.95, zorder=4 if r.precision == MP else 3, **kw)
            if r.acc < YMIN:
                ax.annotate(f"{r.acc:.0f}%", (r.median_ms, y), xytext=(0, -10), textcoords="offset points", ha="center")
        for mk in ["effb0", "mnv3l"]:
            a = d[(d.mk == mk) & (d.precision == "int8_qdq_pct")]
            b = d[(d.mk == mk) & (d.precision == MP)]
            if len(a) and len(b):
                ax.annotate("", xy=(b.median_ms.iloc[0], b.acc.iloc[0]), xytext=(a.median_ms.iloc[0], a.acc.iloc[0]),
                            arrowprops=dict(arrowstyle="->", color=INK, lw=0.7, shrinkA=4, shrinkB=6), zorder=2)
        ax.set_xscale("log")
        ax.set_title(phone + (" (no mixed-precision run)" if phone == "OPPO Reno12 F 4G" else ""))
        ax.set_ylim(YMIN - 1.5, 98.5)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}" if str(int(v))[0] in "25" else ""))
        ax.tick_params(which="minor", labelsize=FS)
        grid(ax, "both")
    for ax in axes[1]:
        ax.set_xlabel("Best median latency on the phone (ms, log scale)")
    for ax in axes[:, 0]:
        ax.set_ylabel("PC test accuracy (%, mean of 3 seeds)")
    hm = [plt.Line2D([], [], ls="", marker="o", ms=6, mfc=mc(m), mec=INK, mew=0.4, label=MODELS[m]) for m in MODELS]
    hp = [plt.Line2D([], [], ls="", marker=MARK[p], ms=5 if p != MP else 8, mfc="#9a9a9a", mec=INK, mew=0.4, label=SHORT[p])
          for p in MARK]
    hp.append(plt.Line2D([], [], ls="", marker="*", ms=8, mfc="white", mec=INK, mew=0.8, label="estimated (†)"))
    hs = [plt.scatter([], [], s=11 * v, color="#bfbfbf", edgecolor=INK, lw=0.4, label=f"{v} MB") for v in (4, 9, 16)]
    r1, r2 = hm + hs, hp + [plt.Line2D([], [], ls="", label=" ")]
    fig.legend(handles=[h for pair in zip(r1, r2) for h in pair], loc="outside upper center", ncol=7)
    save(fig, "Fig5_fig11_tradeoff")


def fig_cross():
    L = pd.read_csv(P / "cross_dataset" / "cross_dataset_long.csv")
    names = {"paddy_doctor": "Paddy Doctor (in-domain)", "sethy": "Sethy et al.", "riceleafbd": "RiceLeafBD"}
    allsets = ["paddy_doctor"] + [d for d in L.dataset.unique() if d != "paddy_doctor"]
    precs = ["int8_dyn", "int8_qdq", "int8_qdq_pct"]
    short = {"int8_dyn": "INT8 dyn", "int8_qdq": "INT8 QDQ", "int8_qdq_pct": "INT8 QDQ-Pct"}
    L["flip_pct"] = 100 * L.flip
    fig, axes = plt.subplots(2, len(allsets), figsize=(W2, 3.7), sharey="row", squeeze=False, layout="constrained")
    w = 0.2
    for c, ds in enumerate(allsets):
        for r, (col, lab) in enumerate([("flip_pct", "Top-1 changed vs FP32 (%)"), ("spearman_all", "CAM Spearman vs FP32")]):
            ax = axes[r][c]
            for k, m in enumerate(MODELS):
                v = [L[(L.dataset == ds) & (L.model == m) & (L.precision == p)][col] for p in precs]
                bar(ax, np.arange(len(precs)) + (k - 1.5) * w, [s.mean() for s in v], w, m, yerr=[s.std() for s in v])
            ax.set_xticks(range(len(precs)), [short[p] for p in precs] if r else [""] * len(precs))
            if c == 0:
                ax.set_ylabel(lab)
            if r == 0:
                ax.set_title(names.get(ds, ds))
            grid(ax)
    axes[1][0].set_ylim(0, 1.02)
    fig.legend(handles=model_handles(), loc="outside upper center", ncol=4)
    save(fig, "Fig6_fig04_cross_dataset")


def fig_latency():
    best = best_all[best_all.precision.isin(PREC_MAIN) & ~best_all.estimated.astype(bool)]
    fig, axes = plt.subplots(1, 4, figsize=(W2, 2.4), layout="constrained")
    w = 0.8 / len(PHONES)
    for ax, m in zip(axes, PMOD):
        for i, ph in enumerate(PHONES):
            v = [best[(best.phone == ph) & (best.model == m) & (best.precision == p)].median_ms.iloc[0] for p in PREC_MAIN]
            x = [k + (i - (len(PHONES) - 1) / 2) * w for k in range(len(PREC_MAIN))]
            bar(ax, x, v, w, color=pc(ph), hatch=SLOT[PSLOT[ph]][2])
        ax.set_xticks(range(len(PREC_MAIN)), ["FP32", "FP16", "INT8\ndyn", "INT8\nQDQ-Pct"])
        ax.set_title(MODELS[PMOD[m]])
        grid(ax)
    axes[0].set_ylabel("Median latency (ms)")
    h = [plt.Rectangle((0, 0), 1, 1, fc=pc(p), hatch=SLOT[PSLOT[p]][2], ec=INK, lw=0.4, label=p) for p in PHONES]
    fig.legend(handles=h, loc="outside upper center", ncol=4)
    save(fig, "S_fig02_latency")


def fig_mixed():
    S = pd.read_csv(P / "mixed_precision" / "mp_summary.csv")
    levels = ["se", "stem", "stem_se", "top1", "top2", "top4", "top8", "top16"]
    order = ["int8_qdq", "int8_qdq_pct"] + [f"mp_{l}" for l in levels]
    lab = {"int8_qdq": "MinMax", "int8_qdq_pct": "Pct", "mp_se": "+SE", "mp_stem": "+stem", "mp_stem_se": "+stem+SE"}
    DS = {"paddy_test": (SLOT[0][0], "o", "-", "Paddy Doctor test"), "sethy": (SLOT[2][0], "s", "--", "Sethy et al."),
          "riceleafbd": (SLOT[1][0], "^", ":", "RiceLeafBD")}
    ms_ = ["effb0", "mnv3l"]
    fig, axes = plt.subplots(2, 2, figsize=(W2, 3.9), sharex="col", layout="constrained")
    for c, m in enumerate(ms_):
        vs = [v for v in order if not S[(S.model == m) & (S.variant == v)].empty]
        xs = np.arange(len(vs))
        for r, (col, yl, sc) in enumerate([("flip", "Top-1 changed vs FP32 (%)", 100), ("spearman", "CAM Spearman vs FP32", 1)]):
            ax = axes[r][c]
            for k, (ds, (co, mk, ls, nm)) in enumerate(DS.items()):
                q = S[(S.model == m) & (S.dataset == ds)].set_index("variant").reindex(vs)
                ax.errorbar(xs + (k - 1) * 0.12, sc * q[f"{col}_mean"], yerr=sc * q[f"{col}_sd"], marker=mk, ls=ls, ms=3,
                            lw=1, capsize=1.2, color=co, mec=INK, mew=0.3, elinewidth=0.6)
            if c == 0:
                ax.set_ylabel(yl)
            grid(ax)
        axes[0][c].set_title(MODELS[m])
        axes[1][c].set_xticks(xs, [lab.get(v, "+" + v[3:]) for v in vs], rotation=35, ha="right")
        axes[1][c].set_xlabel("Static INT8 variant (nodes kept in FP32)")
    h = [plt.Line2D([], [], color=co, marker=mk, ls=ls, ms=3, mec=INK, mew=0.3, label=nm) for co, mk, ls, nm in DS.values()]
    fig.legend(handles=h, loc="outside upper center", ncol=3)
    save(fig, "S_fig05_mixed_precision")


def fig_ood():
    SA = pd.read_csv(P / "ood_guard" / "ood_auroc_summary.csv")
    SC = pd.read_csv(P / "ood_guard" / "confident_summary.csv")
    OOD = {"sethy": "Sethy et al.", "riceleafbd": "RiceLeafBD"}
    SCORES = {"msp": "Max\nsoftmax", "energy": "Energy", "cam_conc": "CAM\nconcentration", "cam_mass": "CAM\nmass"}
    fig, axes = plt.subplots(2, 2, figsize=(W2, 4.6), layout="constrained")
    w = 0.2
    for c, ds in enumerate(OOD):
        ax = axes[0][c]
        for k, m in enumerate(MODELS):
            for j, p in enumerate(["fp32", "int8_qdq_pct"]):
                v = [SA[(SA.model == m) & (SA.precision == p) & (SA.score == s) & (SA.ood == ds)].auroc_mean.iloc[0] for s in SCORES]
                xs = np.arange(len(SCORES)) + (k - 1.5) * w + (j - 0.5) * w * 0.45
                ax.bar(xs, v, w * 0.45, color=mc(m) if j == 0 else tint(mc(m)), edgecolor=INK, linewidth=0.4)
        ax.axhline(0.5, color="#808080", lw=0.6)
        ax.set_ylim(0.3, 1.0)
        ax.set_xticks(range(len(SCORES)), list(SCORES.values()))
        ax.set_title(f"Paddy Doctor vs {OOD[ds]}: AUROC")
        ax.set_ylabel("AUROC")
        grid(ax)
        ax = axes[1][c]
        for m in MODELS:
            for p, ls in (("int8_qdq_pct", "-"), ("int8_qdq", ":")):
                q = SC[(SC.model == m) & (SC.dataset == ds) & (SC.precision == p) & (SC.basis == "fp32")].sort_values("conf")
                ax.plot(q.conf, 100 * q.flip_mean, ls, color=mc(m), lw=1, marker=mm_(m), ms=3, mec=INK, mew=0.3)
        ax.set_xlabel("FP32 top-1 probability cut-off")
        ax.set_ylabel("Top-1 changed vs FP32 (%)")
        ax.set_title(f"{OOD[ds]}: flips on confident images")
        grid(ax)
    h = model_handles("bar") + [plt.Rectangle((0, 0), 1, 1, fc="#808080", ec=INK, lw=0.4, label="FP32 (left bar)"),
                                plt.Rectangle((0, 0), 1, 1, fc=tint("#808080"), ec=INK, lw=0.4, label="INT8 Pct (right, lighter)"),
                                plt.Line2D([], [], color=INK, ls="-", label="INT8 Pct (lines)"),
                                plt.Line2D([], [], color=INK, ls=":", label="INT8 MinMax (lines)")]
    fig.legend(handles=h, loc="outside upper center", ncol=4)
    save(fig, "S_fig07_ood")


def fig_per_class():
    S = pd.read_csv(P / "per_class" / "per_class_summary.csv")
    PRECS = {"int8_qdq": "MinMax", "int8_qdq_pct": "Percentile"}
    SH = {"bacterial_leaf_blight": "BLB", "bacterial_leaf_streak": "BLS", "bacterial_panicle_blight": "BPB", "blast": "Blast",
          "brown_spot": "Brown spot", "dead_heart": "Dead heart", "downy_mildew": "Downy mildew", "hispa": "Hispa",
          "normal": "Normal", "tungro": "Tungro"}
    Dd = S[S.dataset == "paddy_doctor"]
    classes = Dd.sort_values("n", ascending=False).label.unique()
    ms = list(MODELS)
    fig, axes = plt.subplots(1, 4, figsize=(W2, 3.2), layout="constrained")
    for k, (metric, lab, cmap) in enumerate([("flip_mean", "Top-1 flip vs FP32 (%)", "Greys"),
                                             ("spearman_mean", "CAM Spearman vs FP32", "Greys_r")]):
        for j, p in enumerate(PRECS):
            ax = axes[2 * k + j]
            M = np.array([[Dd[(Dd.model == m) & (Dd.label == c) & (Dd.precision == p)][metric].mean() for m in ms] for c in classes])
            if metric == "flip_mean":
                M = 100 * M
                lo, hi = 0, np.nanmax(100 * Dd[Dd.precision.isin(PRECS)].flip_mean)
            else:
                lo, hi = np.nanmin(Dd.spearman_mean), 1
            ax.imshow(M, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
            for (i, jj), v in np.ndenumerate(M):
                t = (v - lo) / (hi - lo)
                dark = t > 0.55 if metric == "flip_mean" else t < 0.45
                ax.text(jj, i, f"{v:.0f}" if metric == "flip_mean" else f"{v:.2f}", ha="center", va="center",
                        color="white" if dark else INK)
            ax.set_xticks(range(len(ms)), ["MNv2", "MNv3-L", "EffL0", "EffB0"], rotation=35, ha="right")
            ax.set_yticks(range(len(classes)), [SH[c] for c in classes] if 2 * k + j == 0 else [""] * len(classes))
            ax.set_title(f"{lab}\n{PRECS[p]}")
            for s in ax.spines.values():
                s.set_visible(False)
            ax.tick_params(length=0)
    save(fig, "S_fig08_per_class")


def fig_calib():
    S = pd.read_csv(F / "_regen" / "calib_size" / "calib_size_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.5), layout="constrained")
    for k, (col, lab, sc) in enumerate([("acc", "Test accuracy (%)", 100), ("flip", "Top-1 changed vs FP32 (%)", 100),
                                        ("spearman", "CAM Spearman vs FP32", 1)]):
        ax = axes[k]
        for m in MODELS:
            for meth, ls in (("minmax", ":"), ("percentile", "-")):
                q = S[(S.model == m) & (S.method == meth)].sort_values("n_calib")
                if q.empty:
                    continue
                ax.errorbar(q.n_calib, sc * q[f"{col}_mean"], yerr=sc * q[f"{col}_sd"].fillna(0), ls=ls, marker=mm_(m), ms=3,
                            lw=1, color=mc(m), capsize=1.2, mec=INK, mew=0.3, elinewidth=0.6)
        ax.set_xscale("log")
        ax.set_xticks([32, 100, 300, 1000], ["32", "100", "300", "1000"])
        ax.minorticks_off()
        ax.set_xlabel("Calibration images")
        ax.set_ylabel(lab)
        grid(ax)
    h = model_handles("line") + [plt.Line2D([], [], color=INK, ls="-", label="Percentile"),
                                 plt.Line2D([], [], color=INK, ls=":", label="MinMax")]
    fig.legend(handles=h, loc="outside upper center", ncol=6)
    save(fig, "S_fig09_calib_size")


if __name__ == "__main__":
    print("font:", font_manager.findfont("Times New Roman"))
    fig_cams("none", ("pdf",))
    fig_cams("antialiased", ("png",))
    fig_insertion()
    fig_layer_error()
    fig_threads()
    fig_tradeoff()
    fig_cross()
    fig_latency()
    fig_mixed()
    fig_ood()
    fig_per_class()
    fig_calib()
