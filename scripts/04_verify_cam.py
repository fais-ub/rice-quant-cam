"""Check that the backprop-free heatmap equals standard Grad-CAM."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from joiv.data import load_split, preprocess_numpy  # noqa: E402
from joiv.model import build_from_checkpoint, cam_from_features  # noqa: E402


def gradcam_autograd(net, x, cls):
    a = net.features(x)
    a.retain_grad()
    logits = net.head(a)
    logits[0, cls].backward()
    alpha = a.grad.mean(dim=(2, 3), keepdim=True)
    return torch.relu((alpha * a).sum(1))[0].detach().numpy()


def r(u, v):
    u, v = u.ravel(), v.ravel()
    if u.std() < 1e-12 or v.std() < 1e-12:
        return float("nan") if not np.allclose(u, v) else 1.0
    return float(np.corrcoef(u, v)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()

    run = Path(a.run)
    net, ck = build_from_checkpoint(str(run / "best.pt"))
    sess = ort.InferenceSession(str(run / "onnx" / "fp32.onnx"), providers=["CPUExecutionProvider"])
    w = net.fc.weight.detach().numpy()
    df = load_split(a.split, "test").sample(frac=1, random_state=0).head(a.n)

    rows = []
    for p in df["path"]:
        xn = preprocess_numpy(p, ck["mean"], ck["std"])
        x = torch.from_numpy(xn)
        with torch.no_grad():
            cls = int(net(x).argmax(1))
        g = gradcam_autograd(net, x.clone().requires_grad_(False), cls)
        with torch.no_grad():
            c_pt = cam_from_features(net.features(x).numpy(), w, [cls])[0]
        logits, feats = sess.run(None, {"image": xn})
        c_ox = cam_from_features(feats, w, [int(logits.argmax(1)[0])])[0]
        rows.append({"r_grad_vs_analytic": r(g, c_pt), "r_torch_vs_onnx": r(c_pt, c_ox),
                     "same_pred": int(logits.argmax(1)[0]) == cls})

    r1 = np.array([x["r_grad_vs_analytic"] for x in rows])
    r2 = np.array([x["r_torch_vs_onnx"] for x in rows])
    res = {"n": len(rows), "min_r_grad_vs_analytic": float(np.nanmin(r1)), "min_r_torch_vs_onnx": float(np.nanmin(r2)),
           "same_pred_rate": float(np.mean([x["same_pred"] for x in rows])),
           "pass": bool(np.nanmin(r1) > 0.999 and np.nanmin(r2) > 0.999)}
    (run / "cam_verification.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
