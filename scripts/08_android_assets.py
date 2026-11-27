"""Copy trained ONNX models and a fixed set of test images into the Android app's assets."""
import argparse
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from joiv.data import load_split  # noqa: E402

PRECISIONS = ["fp32", "fp16", "int8_dyn", "int8_qdq", "int8_qdq_fr", "int8_qdq_pct"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--assets", default=str(Path(__file__).resolve().parents[1] / "android/app/src/main/assets"))
    a = ap.parse_args()

    assets = Path(a.assets)
    for run in map(Path, a.runs):
        d = assets / "models" / run.name
        d.mkdir(parents=True, exist_ok=True)
        for p in PRECISIONS:
            f = run / "onnx" / f"{p}.onnx"
            if f.exists():
                shutil.copy2(f, d / f.name)
        shutil.copy2(run / "onnx" / "head.json", d / "head.json")
        print(run.name, sorted(x.name for x in d.iterdir()))

    te = load_split(a.split, "test")
    per = max(1, a.n // te["label"].nunique())
    sub = te.groupby("label", group_keys=False).sample(n=per, random_state=0).head(a.n).reset_index(drop=True)
    bd = assets / "bench"
    if bd.exists():
        shutil.rmtree(bd)
    bd.mkdir(parents=True)
    rows = []
    for i, r in sub.iterrows():
        name = f"{i:03d}_{r['label']}_{Path(r['path']).stem}.png"
        with Image.open(r["path"]) as im:
            im.convert("RGB").save(bd / name)
        rows.append({"bench": name, "path": r["path"], "label": r["label"]})
    pd.DataFrame(rows).to_csv(assets / "bench_index.csv", index=False)
    print(f"{len(rows)} bench images -> {bd}")


if __name__ == "__main__":
    main()
