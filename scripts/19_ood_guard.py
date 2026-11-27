"""Out-of-domain guard from signals that are free on the phone, and INT8 robustness on confident predictions."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = {"mnv2": "MobileNetV2", "mnv3l": "MobileNetV3-L", "effl0": "EfficientNet-Lite0", "effb0": "EfficientNet-B0"}
PREC = {"fp32": "FP32", "int8_dyn": "INT8 dyn.", "int8_qdq": "INT8 MinMax", "int8_qdq_fr": "INT8 MinMax-FR",
        "int8_qdq_pct": "INT8 Pct."}
SCORES = {"msp": "Max softmax", "neg_entropy": "Neg. entropy", "max_logit": "Max logit", "energy": "Energy",
          "cam_conc": "CAM concentration", "cam_peak": "CAM peak share", "cam_mass": "CAM mass"}
OOD = {"sethy": "Sethy et al.", "riceleafbd": "RiceLeafBD"}
CLASSES = None


def scores(d):
    z = d["logits"].astype(np.float64)
    zm = z.max(1, keepdims=True)
    e = np.exp(z - zm)
    p = e / e.sum(1, keepdims=True)
    cam = d["cam"].reshape(len(z), -1).astype(np.float64)
    mass = cam.sum(1)
    q = np.where(mass[:, None] > 0, cam / np.maximum(mass[:, None], 1e-12), 1.0 / cam.shape[1])
    h = -(q * np.log(np.maximum(q, 1e-12))).sum(1) / np.log(cam.shape[1])
    return pd.DataFrame({"msp": p.max(1), "neg_entropy": (p * np.log(np.maximum(p, 1e-12))).sum(1),
                         "max_logit": z.max(1), "energy": zm[:, 0] + np.log(e.sum(1)),
                         "cam_conc": 1 - h, "cam_peak": q.max(1), "cam_mass": mass})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--precisions", nargs="+", default=["fp32", "int8_dyn", "int8_qdq", "int8_qdq_pct"])
    ap.add_argument("--conf", nargs="+", type=float, default=[0.0, 0.5, 0.7, 0.9, 0.95, 0.99])
    ap.add_argument("--tpr", type=float, default=0.95)
    a = ap.parse_args()
    from sklearn.metrics import roc_auc_score

    dumps, out = Path(a.dumps), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = {ds: pd.read_csv((dumps / "_images" / ds if ds.startswith("paddy") else
                               Path("results/paper_pc/cross_dataset") / ds) / "images.csv", keep_default_na=False).label.values
              for ds in ["paddy_val", "paddy_test", *OOD]}
    import json

    auc_rows, thr_rows, conf_rows = [], [], []
    runs = sorted(p.name for p in (dumps / "paddy_test").iterdir())
    for run in runs:
        classes = json.loads(Path(f"runs/{run}/onnx/head.json").read_text())["classes"]
        cidx = {c: i for i, c in enumerate(classes)}
        D = {}
        for ds in labels:
            for p in a.precisions:
                f = dumps / ds / run / f"{p}.npz"
                if f.exists():
                    D[ds, p] = dict(np.load(f))
        for p in a.precisions:
            if ("paddy_test", p) not in D:
                continue
            sv, st = scores(D["paddy_val", p]), scores(D["paddy_test", p])
            yt = np.array([cidx[l] for l in labels["paddy_test"]])
            pt = D["paddy_test", p]["logits"].argmax(1)
            for s in SCORES:
                thr = float(np.quantile(sv[s], 1 - a.tpr))
                kept = st[s].to_numpy() >= thr
                r0 = {"run": run, "precision": p, "score": s, "threshold": thr, "test_kept": float(kept.mean()),
                      "test_acc_kept": float(np.mean(pt[kept] == yt[kept])), "test_acc_all": float(np.mean(pt == yt))}
                for ds in OOD:
                    if (ds, p) not in D:
                        continue
                    so = scores(D[ds, p])[s].to_numpy()
                    yy = np.r_[np.ones(len(st)), np.zeros(len(so))]
                    auc_rows.append({"run": run, "precision": p, "score": s, "ood": ds,
                                     "auroc": float(roc_auc_score(yy, np.r_[st[s].to_numpy(), so]))})
                    r0[f"fpr_{ds}"] = float(np.mean(so >= thr))
                thr_rows.append(r0)
        for ds in ["paddy_test", *OOD]:
            if (ds, "fp32") not in D:
                continue
            y = np.array([cidx.get(l, -1) for l in labels[ds]])
            ov = sorted(set(y[y >= 0]))
            l32 = D[ds, "fp32"]["logits"]
            p32 = l32.argmax(1)
            c32 = scores(D[ds, "fp32"]).msp.to_numpy()
            for p in a.precisions:
                if p == "fp32" or (ds, p) not in D:
                    continue
                d = D[ds, p]
                pp = d["logits"].argmax(1)
                cp = scores(d).msp.to_numpy()
                for basis, conf in (("fp32", c32), ("own", cp), ("both", np.minimum(c32, cp))):
                    for c in a.conf:
                        m = conf >= c
                        if m.sum() == 0:
                            continue
                        conf_rows.append({"run": run, "dataset": ds, "precision": p, "basis": basis, "conf": c,
                                          "n": int(m.sum()), "frac": float(m.mean()),
                                          "acc_fp32": float(np.mean(p32[m] == y[m])),
                                          "acc_ov_fp32": float(np.mean(np.array(ov)[l32[m][:, ov].argmax(1)] == y[m])),
                                          "acc": float(np.mean(pp[m] == y[m])),
                                          "flip": float(np.mean(pp[m] != p32[m])),
                                          "spearman": float(np.nanmean(d["spearman"][m])),
                                          "iou_top20": float(np.nanmean(d["iou_top20"][m]))})
        print(run, flush=True)

    def summ(df, keys, cols):
        df = df.assign(model=df.run.str.split("_s").str[0])
        g = df.groupby(["model", *keys], sort=False)
        s = g[cols].mean().add_suffix("_mean").join(g[cols].std().add_suffix("_sd"))
        s["n_seeds"] = g.size()
        return s.reset_index()

    A, T, C = pd.DataFrame(auc_rows), pd.DataFrame(thr_rows), pd.DataFrame(conf_rows)
    A.to_csv(out / "ood_auroc_long.csv", index=False)
    T.to_csv(out / "ood_threshold_long.csv", index=False)
    C.to_csv(out / "confident_long.csv", index=False)
    SA = summ(A, ["precision", "score", "ood"], ["auroc"])
    ST = summ(T, ["precision", "score"], [c for c in T.columns if c not in ("run", "precision", "score")])
    SC = summ(C, ["dataset", "precision", "basis", "conf"], ["n", "frac", "acc_fp32", "acc_ov_fp32", "acc", "flip",
                                                              "spearman", "iou_top20"])
    SA.to_csv(out / "ood_auroc_summary.csv", index=False)
    ST.to_csv(out / "ood_threshold_summary.csv", index=False)
    SC.to_csv(out / "confident_summary.csv", index=False)

    def ms(v, sd, d=3, scale=1.0):
        return "--" if pd.isna(v) else f"{scale * v:.{d}f}{{\\scriptsize$\\pm${scale * sd:.{d}f}}}"

    main_scores = ["msp", "neg_entropy", "energy", "cam_conc", "cam_mass"]
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Out-of-domain guard from signals the phone already has. AUROC of Paddy Doctor test images vs each "
             r"external set, and the share of external images let through (FPR) at the threshold that keeps "
             rf"{int(100 * a.tpr)}\,\% of Paddy Doctor validation images. Mean $\pm$ sd over 3 seeds.}}",
             r"\label{tab:ood}", r"\begin{tabular}{lll" + "rr" * len(OOD) + "r}", r"\toprule",
             r"Model & Precision & Score & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{v}}}" for v in OOD.values()) +
             r" & Test kept \\",
             "".join(rf"\cmidrule(lr){{{4 + 2 * i}-{5 + 2 * i}}}" for i in range(len(OOD))),
             r" & & & " + " & ".join(["AUROC", r"FPR (\%)"] * len(OOD)) + r" & (\%) \\", r"\midrule"]
    for m in MODELS:
        for p in ["fp32", "int8_qdq_pct"]:
            for k, s in enumerate(main_scores):
                cells = []
                for ds in OOD:
                    q = SA[(SA.model == m) & (SA.precision == p) & (SA.score == s) & (SA.ood == ds)]
                    t = ST[(ST.model == m) & (ST.precision == p) & (ST.score == s)]
                    cells += [ms(q.auroc_mean.iloc[0], q.auroc_sd.iloc[0]),
                              ms(t[f"fpr_{ds}_mean"].iloc[0], t[f"fpr_{ds}_sd"].iloc[0], 1, 100)]
                t = ST[(ST.model == m) & (ST.precision == p) & (ST.score == s)]
                lead = (MODELS[m] if p == "fp32" and k == 0 else "") + " & " + (PREC[p] if k == 0 else "")
                lines.append(f"{lead} & {SCORES[s]} & " + " & ".join(cells) + f" & {100 * t.test_kept_mean.iloc[0]:.1f}" + r" \\")
            lines.append(r"\cmidrule(l){2-" + str(4 + 2 * len(OOD)) + "}" if p == "fp32" else r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}"]
    (out / "tab_ood.tex").write_text("\n".join(lines) + "\n")

    cut = 0.9
    sets = ["paddy_test", *OOD]
    nm = {"paddy_test": "Paddy Doctor", **OOD}
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{INT8 vs FP32 on all images and on images whose FP32 top-1 probability is $\ge " + f"{cut}" +
             r"$ (share of images in that subset in brackets). Flip (\%) and Grad-CAM Spearman $\rho$ vs FP32, "
             r"mean over 3 seeds.}", r"\label{tab:confident}",
             r"\begin{tabular}{ll" + "rrrr" * len(sets) + "}", r"\toprule",
             r" & & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{nm[d]}}}" for d in sets) + r" \\",
             "".join(rf"\cmidrule(lr){{{3 + 4 * i}-{6 + 4 * i}}}" for i in range(len(sets))),
             r" & & " + " & ".join([r"\multicolumn{2}{c}{All}", rf"\multicolumn{{2}}{{c}}{{Conf.\ $\ge {cut}$}}"] * len(sets)) + r" \\",
             r"Model & Precision & " + " & ".join(["Flip", r"$\rho$"] * 2 * len(sets)) + r" \\", r"\midrule"]
    for m in MODELS:
        for j, p in enumerate(["int8_qdq", "int8_qdq_pct"]):
            cells = []
            for ds in sets:
                for c in (0.0, cut):
                    q = SC[(SC.model == m) & (SC.dataset == ds) & (SC.precision == p) & (SC.basis == "fp32") & (SC.conf == c)]
                    if q.empty:
                        cells += ["--", "--"]
                        continue
                    r = q.iloc[0]
                    fl = f"{100 * r.flip_mean:.1f}" + (f" ({100 * r.frac_mean:.0f}\\%)" if c > 0 else "")
                    cells += [fl, f"{r.spearman_mean:.2f}"]
            lines.append(f"{MODELS[m] if j == 0 else ''} & {PREC[p]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}"]
    (out / "tab_confident.tex").write_text("\n".join(lines) + "\n")
    pd.set_option("display.width", 250)
    print(SA[SA.precision.isin(["fp32", "int8_qdq_pct"])].pivot_table(index=["model", "precision"], columns=["ood", "score"],
                                                                      values="auroc_mean").round(3).to_string())
    print(ST[ST.precision.isin(["fp32", "int8_qdq_pct"]) & ST.score.isin(["msp", "energy", "cam_conc"])]
          [["model", "precision", "score", "test_kept_mean", "fpr_sethy_mean", "fpr_riceleafbd_mean", "test_acc_kept_mean",
            "test_acc_all_mean"]].round(3).to_string(index=False))
    print(SC[(SC.basis == "fp32") & SC.conf.isin([0.0, 0.9])][["model", "dataset", "precision", "conf", "frac_mean",
                                                               "acc_fp32_mean", "flip_mean", "spearman_mean"]]
          .round(3).to_string(index=False))


def figure(out):
    """fig_ood: AUROC per score (FP32 vs INT8 Pct) and flip vs FP32 confidence cut-off."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    SA = pd.read_csv(out / "ood_auroc_summary.csv")
    SC = pd.read_csv(out / "confident_summary.csv")
    MCOL = {"mnv2": "#2a78d6", "mnv3l": "#eb6834", "effl0": "#1baf7a", "effb0": "#eda100"}
    sc = ["msp", "energy", "cam_conc", "cam_mass"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    w = 0.2
    for c, ds in enumerate(OOD):
        ax = axes[0][c]
        for k, m in enumerate(MODELS):
            for j, p in enumerate(["fp32", "int8_qdq_pct"]):
                v = [SA[(SA.model == m) & (SA.precision == p) & (SA.score == s) & (SA.ood == ds)].auroc_mean.iloc[0] for s in sc]
                xs = np.arange(len(sc)) + (k - 1.5) * w
                ax.bar(xs, v, w * (0.45 if j else 0.9), color=MCOL[m], alpha=1.0 if j == 0 else 0.45,
                       label=MODELS[m] if j == 0 else None, align="edge" if j else "center")
        ax.axhline(0.5, color="#9a9994", lw=0.6)
        ax.set_ylim(0.3, 1.0)
        ax.set_xticks(range(len(sc)), [SCORES[s].replace(" ", "\n") for s in sc], fontsize=6.5)
        ax.set_title(f"Paddy Doctor vs {OOD[ds]}: AUROC\n(solid FP32, light INT8 Pct.)", fontsize=7.5)
        ax = axes[1][c]
        for m in MODELS:
            for p, ls in (("int8_qdq_pct", "-"), ("int8_qdq", ":")):
                q = SC[(SC.model == m) & (SC.dataset == ds) & (SC.precision == p) & (SC.basis == "fp32")].sort_values("conf")
                ax.plot(q.conf, 100 * q.flip_mean, ls, color=MCOL[m], lw=1.1, marker="o", ms=2.5)
        ax.set_xlabel("FP32 top-1 probability cut-off", fontsize=7)
        ax.set_ylabel("Top-1 changed vs FP32 (%)", fontsize=7)
        ax.set_title(f"{OOD[ds]}: flips on confident images\n(solid Pct., dotted MinMax)", fontsize=7.5)
        for a_ in axes[:, c]:
            a_.tick_params(labelsize=6.5, colors="#52514e")
            for s in ["top", "right"]:
                a_.spines[s].set_visible(False)
    axes[0][0].legend(fontsize=6.5, frameon=False, ncol=2)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(out / f"fig_ood.{ext}", dpi=300)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2 and sys.argv[1] == "figure":
        figure(sys.argv[2])
    else:
        main()
