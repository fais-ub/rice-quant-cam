"""Evaluate all ONNX precisions of one run on the test set (on the PC)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from joiv.data import load_split, preprocess_numpy  # noqa: E402
from joiv.model import cam_from_features  # noqa: E402
from joiv.xai_metrics import deletion_insertion, is_degenerate, localisation, normalise, stability, upsample  # noqa: E402

PRECISIONS = ["fp32", "fp16", "int8_dyn", "int8_qdq", "int8_qdq_fr", "int8_qdq_pct"]


def mcnemar_p(correct_a, correct_b):
    from scipy.stats import binomtest

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    return 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--subset", default="test")
    ap.add_argument("--masks", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--faith-n", type=int, default=300, help="test images used for deletion/insertion (0 = off)")
    ap.add_argument("--faith-steps", type=int, default=10)
    a = ap.parse_args()

    run = Path(a.run)
    head = json.loads((run / "onnx" / "head.json").read_text())
    classes, w = head["classes"], np.array(head["fc_weight"], dtype=np.float32)
    df = load_split(a.split, a.subset)
    if a.limit:
        df = df.sample(min(a.limit, len(df)), random_state=0).reset_index(drop=True)
    y = np.array([classes.index(l) for l in df["label"]])

    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    sessions = {p: ort.InferenceSession(str(run / "onnx" / f"{p}.onnx"), so, providers=["CPUExecutionProvider"])
                for p in PRECISIONS if (run / "onnx" / f"{p}.onnx").exists()}

    preds = {p: [] for p in sessions}
    cams = {p: [] for p in sessions}
    for path in df["path"]:
        x = preprocess_numpy(path, head["mean"], head["std"])
        for p, s in sessions.items():
            logits, feats = s.run(None, {"image": x})
            c = int(logits.argmax(1)[0])
            preds[p].append(c)
            cams[p].append(cam_from_features(feats, w, [c])[0])

    summary, per_image = [], []
    ref_ok = np.array(preds["fp32"]) == y
    for p in sessions:
        pr = np.array(preds[p])
        row = {"precision": p, "n": len(y), "acc": accuracy_score(y, pr), "macro_f1": f1_score(y, pr, average="macro"),
               "flip_rate_vs_fp32": float(np.mean(pr != np.array(preds["fp32"]))),
               "mcnemar_p_vs_fp32": mcnemar_p(ref_ok, pr == y),
               "degenerate_cam_rate": float(np.mean([is_degenerate(c) for c in cams[p]]))}
        for k, v in zip(classes, f1_score(y, pr, average=None, labels=range(len(classes)))):
            row[f"f1_{k}"] = float(v)
        summary.append(row)

    for i, path in enumerate(df["path"]):
        ref = normalise(upsample(cams["fp32"][i]))
        mask = None
        if a.masks:
            mp = Path(a.masks) / (Path(path).stem + ".png")
            if mp.exists():
                with Image.open(mp) as m:
                    mask = np.asarray(m.convert("L").resize((224, 224), Image.NEAREST)) > 127
        for p in sessions:
            hm = normalise(upsample(cams[p][i]))
            rec = {"path": path, "label": df["label"][i], "precision": p, "pred": classes[preds[p][i]],
                   "same_pred_as_fp32": preds[p][i] == preds["fp32"][i]}
            if p != "fp32":
                rec.update(stability(ref, hm))
            if mask is not None and mask.any():
                rec.update(localisation(hm, mask))
            per_image.append(rec)

    faith_idx = set(range(min(a.faith_n, len(df))))
    rng = np.random.default_rng(0)
    for i in sorted(faith_idx):
        x = preprocess_numpy(df["path"][i], head["mean"], head["std"])
        for p, s in sessions.items():
            pred_fn = (lambda s_: (lambda xx: s_.run(["logits"], {"image": xx})[0]))(s)
            hm = upsample(cams[p][i])
            res = deletion_insertion(x, hm, preds[p][i], pred_fn, a.faith_steps)
            if p == "fp32":
                rmap = upsample(rng.random(cams[p][i].shape).astype(np.float32))
                rnd = deletion_insertion(x, rmap, preds[p][i], pred_fn, a.faith_steps)
                res.update({"deletion_auc_random": rnd["deletion_auc"], "insertion_auc_random": rnd["insertion_auc"]})
            for rec in per_image:
                if rec["path"] == df["path"][i] and rec["precision"] == p:
                    rec.update(res)
                    break

    pi = pd.DataFrame(per_image)
    metric_cols = [c for c in pi.columns if c.startswith(("spearman", "ssim", "iou_", "energy_", "pointing_", "deletion_", "insertion_"))]
    agg = []
    for p, g in pi.groupby("precision", sort=False):
        for scope, gg in (("all", g), ("same_pred", g[g["same_pred_as_fp32"]])):
            agg.append({"precision": p, "scope": scope, "n": len(gg), **gg[metric_cols].mean(numeric_only=True).to_dict()})

    ed = run / "eval"
    ed.mkdir(exist_ok=True)
    pd.DataFrame(summary).to_csv(ed / f"{a.subset}_classification.csv", index=False)
    pi.to_csv(ed / f"{a.subset}_per_image.csv", index=False)
    pd.DataFrame(agg).to_csv(ed / f"{a.subset}_heatmap_summary.csv", index=False)
    print(pd.DataFrame(summary)[["precision", "acc", "macro_f1", "flip_rate_vs_fp32", "degenerate_cam_rate"]].to_string(index=False))
    print(pd.DataFrame(agg).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
