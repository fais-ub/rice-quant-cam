"""Ablation of INT8 QDQ settings: activation type (U8S8 vs S8S8) x calibration method."""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from joiv.data import load_split, preprocess_numpy  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

CalibReader = import_module("03_export_onnx").CalibReader


class _ListReader:
    def __init__(self, items):
        self.it = iter(items)

    def get_next(self):
        return next(self.it, None)


def _chunk_histogram_calibration(chunk=20):
    from onnxruntime.quantization.calibrate import HistogramCalibrater

    orig = HistogramCalibrater.collect_data
    if getattr(orig, "_chunked", False):
        return

    def collect_data(self, data_reader):
        while True:
            batch = []
            while len(batch) < chunk:
                x = data_reader.get_next()
                if not x:
                    break
                batch.append(x)
            if not batch:
                break
            orig(self, _ListReader(batch))

    collect_data._chunked = True
    HistogramCalibrater.collect_data = collect_data


def predict(path, xs):
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return np.array([int(s.run(["logits"], {"image": x})[0].argmax()) for x in xs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int, default=0, help="test images to use (0 = all)")
    a = ap.parse_args()

    import json

    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    _chunk_histogram_calibration()
    run = Path(a.run)
    od = run / "onnx"
    head = json.loads((od / "head.json").read_text())
    classes = head["classes"]
    te = load_split(a.split, "test")
    if a.limit:
        te = te.sample(min(a.limit, len(te)), random_state=0).reset_index(drop=True)
    xs = [preprocess_numpy(p, head["mean"], head["std"]) for p in te["path"]]
    y = np.array([classes.index(l) for l in te["label"]])
    tr = load_split(a.split, "train")
    calib = tr[tr["calib"]]["path"].tolist()

    abl = od / "ablation"
    abl.mkdir(exist_ok=True)
    pre = abl / "_pre.onnx"
    quant_pre_process(str(od / "fp32.onnx"), str(pre))

    ref = predict(od / "fp32.onnx", xs)
    rows = [{"variant": "fp32", "acc": accuracy_score(y, ref), "macro_f1": f1_score(y, ref, average="macro"), "flip_vs_fp32": 0.0}]
    methods = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile,
               "entropy": CalibrationMethod.Entropy}
    for act, qt in (("u8", QuantType.QUInt8), ("s8", QuantType.QInt8)):
        for mname, m in methods.items():
            for reduce in (False, True):
                tag = f"{act}_{mname}{'_rr' if reduce else ''}"
                out = abl / f"int8_qdq_{tag}.onnx"
                try:
                    quantize_static(str(pre), str(out), CalibReader(calib, head["mean"], head["std"]),
                                    quant_format=QuantFormat.QDQ, per_channel=True, reduce_range=reduce,
                                    weight_type=QuantType.QInt8, activation_type=qt, calibrate_method=m)
                    pr = predict(out, xs)
                    rows.append({"variant": tag, "acc": accuracy_score(y, pr), "macro_f1": f1_score(y, pr, average="macro"),
                                 "flip_vs_fp32": float(np.mean(pr != ref))})
                except Exception as e:
                    rows.append({"variant": tag, "error": str(e)[:200]})
                print(rows[-1], flush=True)
    pre.unlink(missing_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(abl / "qdq_ablation.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
