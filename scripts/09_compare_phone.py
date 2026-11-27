"""Fidelity check: do phone predictions and CAMs match the PC for the same ONNX file?"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from joiv.data import preprocess_numpy  # noqa: E402
from joiv.model import cam_from_features  # noqa: E402
from joiv.xai_metrics import upsample  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--assets", required=True)
    a = ap.parse_args()

    ph = pd.read_csv(a.bench)
    bench_dir = Path(a.assets) / "bench"
    rows = []
    for (model, prec), g in ph.groupby(["model", "precision"]):
        od = Path(a.runs_root) / model / "onnx"
        head = json.loads((od / "head.json").read_text())
        w = np.array(head["fc_weight"], dtype=np.float32)
        s = ort.InferenceSession(str(od / f"{prec}.onnx"), providers=["CPUExecutionProvider"])
        pc = {}
        for img in g["image"].unique():
            logits, feats = s.run(None, {"image": preprocess_numpy(str(bench_dir / img), head["mean"], head["std"])})
            c = int(logits.argmax())
            pc[img] = (head["classes"][c], cam_from_features(feats, w, [c])[0])
        for ep, gg in g.groupby("ep"):
            agree, maxdiff, rho = [], [], []
            for _, r in gg.iterrows():
                cls, cam = pc[r["image"]]
                phone_cam = np.array([float(v) for v in str(r["cam"]).split()], dtype=np.float32).reshape(cam.shape)
                agree.append(cls == r["pred"])
                maxdiff.append(float(np.abs(phone_cam - cam).max()))
                u1, u2 = upsample(cam).ravel(), upsample(phone_cam).ravel()
                rho.append(spearmanr(u1, u2).statistic if u1.std() > 0 and u2.std() > 0 else np.nan)
            rows.append({"device": gg["device"].iloc[0], "model": model, "precision": prec, "ep": ep, "n": len(gg),
                         "pred_agreement": np.mean(agree), "cam_maxabs_diff": np.max(maxdiff),
                         "cam_spearman_mean": np.nanmean(rho)})
    out = pd.DataFrame(rows)
    out.to_csv(Path(a.bench).with_name("fidelity_vs_pc.csv"), index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
