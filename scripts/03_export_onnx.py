"""Export a trained checkpoint to the four precisions used in the paper."""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from joiv.data import load_split, preprocess_numpy  # noqa: E402
from joiv.model import ExportWrapper, build_from_checkpoint  # noqa: E402


class HalfWrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = copy.deepcopy(net).half().eval()

    def forward(self, x):
        a = self.net.features(x.half())
        return self.net.head(a).float(), a.float()


class CalibReader:
    def __init__(self, paths, mean, std, name="image"):
        self.it = iter(paths)
        self.mean, self.std, self.name = mean, std, name

    def get_next(self):
        p = next(self.it, None)
        return None if p is None else {self.name: preprocess_numpy(p, self.mean, self.std)}

    def rewind(self):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--act", choices=["s8", "u8"], default="s8", help="activation type for int8_qdq")
    a = ap.parse_args()

    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_dynamic, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    run = Path(a.run)
    net, ck = build_from_checkpoint(str(run / "best.pt"))
    od = run / "onnx"
    od.mkdir(exist_ok=True)

    fp32 = od / "fp32.onnx"
    torch.onnx.export(ExportWrapper(net), torch.zeros(1, 3, 224, 224), str(fp32), dynamo=False,
                      input_names=["image"], output_names=["logits", "features"], opset_version=a.opset)
    onnx.checker.check_model(str(fp32))

    torch.onnx.export(HalfWrapper(net), torch.zeros(1, 3, 224, 224), str(od / "fp16.onnx"), dynamo=False,
                      input_names=["image"], output_names=["logits", "features"], opset_version=a.opset)

    pre = od / "_pre.onnx"
    quant_pre_process(str(fp32), str(pre))

    quantize_dynamic(str(pre), str(od / "int8_dyn.onnx"), weight_type=QuantType.QUInt8)

    df = load_split(a.split, "train")
    calib = df[df["calib"]]["path"].tolist() if "calib" in df else df["path"].tolist()[:300]
    act = QuantType.QUInt8 if a.act == "u8" else QuantType.QInt8
    for name, rr in (("int8_qdq.onnx", True), ("int8_qdq_fr.onnx", False)):
        quantize_static(str(pre), str(od / name), CalibReader(calib, ck["mean"], ck["std"]),
                        quant_format=QuantFormat.QDQ, per_channel=True, reduce_range=rr,
                        weight_type=QuantType.QInt8, activation_type=act,
                        calibrate_method=CalibrationMethod.MinMax)
    pre.unlink(missing_ok=True)

    head = {
        "arch": ck["arch"], "timm_name": ck["timm_name"], "classes": ck["classes"],
        "mean": list(ck["mean"]), "std": list(ck["std"]), "input": [1, 3, 224, 224],
        "fc_weight": net.fc.weight.detach().numpy().round(7).tolist(),
        "fc_bias": net.fc.bias.detach().numpy().round(7).tolist(),
    }
    (od / "head.json").write_text(json.dumps(head))
    for f in sorted(od.glob("*.onnx")):
        print(f"{f.name:16s} {f.stat().st_size / 1e6:7.2f} MB")


if __name__ == "__main__":
    main()
