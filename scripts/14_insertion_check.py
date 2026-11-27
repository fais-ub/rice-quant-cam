"""Map-swap check: does a higher INT8 insertion AUC come from the heatmap or from the scoring model?"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from joiv.data import load_split, preprocess_numpy  # noqa: E402
from joiv.model import cam_from_features  # noqa: E402
from joiv.xai_metrics import upsample  # noqa: E402

PRECISIONS = ["fp32", "fp16", "int8_dyn", "int8_qdq", "int8_qdq_fr", "int8_qdq_pct"]
TRAP = getattr(np, "trapezoid", None) or np.trapz


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def curves(x, blurred, sal, cls, logits_fn, steps):
    """Same procedure as joiv.xai_metrics.deletion_insertion, but batched and returning the curves."""
    h, w = sal.shape
    order = np.argsort(-sal.ravel(), kind="stable")
    fr = np.linspace(0, 1, steps + 1)
    xd, xi = [], []
    for f in fr:
        k = int(round(f * h * w))
        m = np.zeros(h * w, dtype=bool)
        m[order[:k]] = True
        m = m.reshape(1, 1, h, w)
        xd.append(np.where(m, 0.0, x)[0])
        xi.append(np.where(m, x, blurred)[0])
    pd_ = softmax(logits_fn(np.stack(xd).astype(np.float32)))[:, cls]
    pi_ = softmax(logits_fn(np.stack(xi).astype(np.float32)))[:, cls]
    return pd_, pi_, float(TRAP(pd_, fr)), float(TRAP(pi_, fr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--faith-n", type=int, default=300)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    run = Path(a.run)
    head = json.loads((run / "onnx" / "head.json").read_text())
    classes, w = head["classes"], np.array(head["fc_weight"], dtype=np.float32)
    df = load_split(a.split, "test").iloc[: a.faith_n].reset_index(drop=True)

    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    sess = {p: ort.InferenceSession(str(run / "onnx" / f"{p}.onnx"), so, providers=["CPUExecutionProvider"])
            for p in PRECISIONS if (run / "onnx" / f"{p}.onnx").exists()}

    def logits_of(p):
        s = sess[p]
        return lambda xx: np.concatenate([s.run(["logits"], {"image": xx[j:j + 1]})[0] for j in range(len(xx))])

    fns = {p: logits_of(p) for p in sess}
    rows, curv = [], {}
    t0 = time.time()
    for i, path in enumerate(df["path"]):
        x = preprocess_numpy(path, head["mean"], head["std"])
        blurred = np.stack([gaussian_filter(x[0, c], 10.0) for c in range(3)])[None].astype(np.float32)
        zero = np.zeros_like(x)
        out = {p: sess[p].run(None, {"image": x}) for p in sess}
        c32 = int(out["fp32"][0].argmax())
        cam32_shape = out["fp32"][1].shape[2:]
        rmap = upsample(np.random.default_rng(1000 + i).random(cam32_shape).astype(np.float32))
        for p in sess:
            lg, ft = out[p]
            c = int(lg.argmax())
            pr = softmax(lg[0])
            own = upsample(cam_from_features(ft, w, [c])[0])
            fp32map = upsample(cam_from_features(out["fp32"][1], w, [c])[0])
            rec = {"idx": i, "path": path, "label": df["label"][i], "precision": p, "pred": classes[c],
                   "correct": classes[c] == df["label"][i], "same_pred_as_fp32": c == c32,
                   "conf_full": float(pr[c]), "maxprob": float(pr.max()),
                   "entropy": float(-(pr * np.log(pr + 1e-12)).sum()),
                   "logit_margin": float(np.sort(lg[0])[-1] - np.sort(lg[0])[-2]),
                   "conf_blur": float(softmax(fns[p](blurred)[0])[c]),
                   "conf_zero": float(softmax(fns[p](zero)[0])[c])}
            for name, sal in (("own", own), ("fp32map", fp32map), ("rand", rmap)):
                dc, ic, da, ia = curves(x, blurred, sal, c, fns[p], a.steps)
                rec[f"del_{name}"], rec[f"ins_{name}"] = da, ia
                curv[(p, name, i)] = (dc, ic)
            if p != "fp32" and c == c32:
                dc, ic, da, ia = curves(x, blurred, own, c32, fns["fp32"], a.steps)
                rec["del_own_at_fp32"], rec["ins_own_at_fp32"] = da, ia
            rows.append(rec)
        if i % 25 == 0:
            print(f"{run.name} {i}/{len(df)} {time.time() - t0:.0f}s", flush=True)

    res = pd.DataFrame(rows)
    ed = run / "eval"
    res.to_csv(ed / "insertion_check.csv", index=False)
    keys = sorted({(p, n) for p, n, _ in curv})
    np.savez_compressed(ed / "insertion_check_curves.npz",
                        **{f"{p}__{n}__{k}": np.stack([curv[(p, n, i)][j] for i in range(len(df))])
                           for p, n in keys for j, k in enumerate(["del", "ins"])})
    print(res.groupby("precision", sort=False)[["conf_full", "conf_blur", "ins_own", "ins_rand", "ins_fp32map"]]
          .mean().round(3).to_string())
    print(f"done {run.name} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
