#!/usr/bin/env bash
# Requires a Kaggle API token and the accepted rules of the paddy-disease-classification competition.
set -euo pipefail
OUT=${1:-data/raw/paddy}
mkdir -p "$OUT"
if [ -z "${KAGGLE_API_TOKEN:-}" ] && [ -f "$HOME/.kaggle/kaggle.json" ]; then
  KAGGLE_API_TOKEN="$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))["key"])')"
  export KAGGLE_API_TOKEN
fi
kaggle competitions download -c paddy-disease-classification -p "$OUT"
unzip -q -o "$OUT/paddy-disease-classification.zip" -d "$OUT" && rm "$OUT/paddy-disease-classification.zip"
echo "Labelled images: $OUT/train_images/<label>/"
