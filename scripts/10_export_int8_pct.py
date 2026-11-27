"""Add a percentile-calibrated static INT8 model to an exported run: <run>/onnx/int8_qdq_pct.onnx."""
import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from joiv.data import load_split  # noqa: E402

CalibReader = import_module("03_export_onnx").CalibReader
_chunk_histogram_calibration = import_module("07_qdq_ablation")._chunk_histogram_calibration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--act", choices=["s8", "u8"], default="s8")
    a = ap.parse_args()

    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    _chunk_histogram_calibration()
    od = Path(a.run) / "onnx"
    head = json.loads((od / "head.json").read_text())
    tr = load_split(a.split, "train")
    calib = tr[tr["calib"]]["path"].tolist() if "calib" in tr else tr["path"].tolist()[:300]
    pre = od / "_pre_pct.onnx"
    quant_pre_process(str(od / "fp32.onnx"), str(pre))
    out = od / "int8_qdq_pct.onnx"
    quantize_static(str(pre), str(out), CalibReader(calib, head["mean"], head["std"]),
                    quant_format=QuantFormat.QDQ, per_channel=True, reduce_range=False,
                    weight_type=QuantType.QInt8,
                    activation_type=QuantType.QUInt8 if a.act == "u8" else QuantType.QInt8,
                    calibrate_method=CalibrationMethod.Percentile)
    pre.unlink(missing_ok=True)
    print(f"{out}  {out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
