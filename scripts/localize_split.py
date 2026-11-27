"""Turn the published split (relative paths) into one with absolute paths for this machine."""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=str(Path(__file__).resolve().parents[1] / "data/splits/paddy.csv"))
    ap.add_argument("--root", required=True, help="folder that contains train_images/")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    root = Path(a.root).resolve()
    df = pd.read_csv(a.split)
    df["path"] = [str(root / p) for p in df["path"]]
    missing = [p for p in df["path"] if not Path(p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images not found, e.g. {missing[0]}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} images -> {a.out}")


if __name__ == "__main__":
    main()
