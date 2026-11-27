"""Scan, de-duplicate and split the dataset."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from joiv.data import add_hashes, dedup, scan_class_folders, stratified_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing <label>/<image> folders")
    ap.add_argument("--out", required=True, help="output CSV with path,label,split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--calib", type=int, default=300, help="calibration images for INT8, taken from train")
    ap.add_argument("--exclude", nargs="*", default=[], help="labels to leave out")
    a = ap.parse_args()

    df = scan_class_folders(a.root)
    df = df[~df["label"].isin(a.exclude)]
    print(f"found {len(df)} images, {df['label'].nunique()} classes")
    df = add_hashes(df)
    kept, rep = dedup(df)
    print(f"after de-dup: {len(kept)} images ({len(df) - len(kept)} removed)")

    out = stratified_split(kept, a.val, a.test, a.seed)
    calib = out[out["split"] == "train"].sample(min(a.calib, (out["split"] == "train").sum()), random_state=a.seed)
    out["calib"] = out.index.isin(calib.index)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out[["path", "label", "split", "calib", "md5", "phash"]].to_csv(a.out, index=False)
    rep.to_csv(Path(a.out).with_suffix(".dedup_report.csv"), index=False)
    summary = out.pivot_table(index="label", columns="split", values="path", aggfunc="count", fill_value=0)
    summary["total"] = summary.sum(axis=1)
    summary.loc["TOTAL"] = summary.sum()
    summary.to_csv(Path(a.out).with_suffix(".summary.csv"))
    print(summary.to_string())


if __name__ == "__main__":
    main()
