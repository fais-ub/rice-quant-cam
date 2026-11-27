# Quantization effects on rice disease predictions and CAMs on Android phones

Code for the paper *Quantization Effects on Rice Disease Predictions and Class Activation Maps on Android Phones*
(F. A. Huda, J. Muchlisyiyah, H. Tolle, F. Utaminingrum).

Four ImageNet-pretrained backbones (MobileNetV2, MobileNetV3-Large, EfficientNet-Lite0, EfficientNet-B0) are
fine-tuned on the ten-class Paddy Doctor subset, exported to ONNX in six precisions (FP32, FP16, dynamic INT8 and
three static QDQ INT8 variants) plus a sensitivity-guided mixed-precision variant, and evaluated for accuracy,
prediction flips, CAM stability, deletion/insertion faithfulness, PC–phone agreement and latency on four Android
phones with ONNX Runtime.

This code release (v1.0.0) is archived on Zenodo: DOI [10.5281/zenodo.24575665](https://doi.org/10.5281/zenodo.24575665).
Trained checkpoints, ONNX models and raw experiment records are not included.

## Layout

```
src/joiv/        model (GAP + linear head), data loading, heatmap metrics
scripts/         numbered pipeline steps, paper tables and figures
android/         benchmark app (Kotlin, Jetpack Compose, ONNX Runtime Android 1.32.0)
data/            split manifest, external class mapping, list of phone benchmark images
```

The classifier head is global average pooling followed by one linear layer, so Grad-CAM for class *c* reduces to
ReLU(Σ_k w_ck A_k)/(HW). Every exported model returns both `logits` and the last feature map `features`; the phone
builds the CAM from `features` and the FP32 weights in `head.json` without backpropagation.
`scripts/04_verify_cam.py` checks this against autograd Grad-CAM.

## Setup

Experiments ran on Ubuntu 24.04 (WSL2) with Python 3.12, CUDA 13.0 and an RTX 5080.

```bash
bash scripts/00_setup_env.sh                       # creates ~/venvs/joiv
source ~/venvs/joiv/bin/activate
```

Plotting scripts (`11`, `12`, `13 plot`, `15 summarize`, `17 summarize`, `18 figure`, `20`, `21 figure`, `23`, `24`)
were run in a second environment with `requirements-plot.txt`. Exact versions are in `requirements.lock.txt`.

## Data

Paddy Doctor images come from the Kaggle competition `paddy-disease-classification` (accept the rules first):

```bash
bash scripts/get_data.sh ~/joiv-data/raw/paddy
python scripts/localize_split.py --root ~/joiv-data/raw/paddy --out ~/joiv-data/splits/paddy.csv
```

`data/splits/paddy.csv` is the split used in the paper (10,201 images after MD5 and perceptual-hash
de-duplication; 7,140 / 1,530 / 1,531, seed 42, with the 300 calibration images marked). `01_prepare_data.py`
rebuilds it from the raw images.

External test sets, used without retraining:

- Sethy, *Rice Leaf Disease Image Samples*, Mendeley Data, doi:10.17632/fwcj7stb8r.1
- Rimi and Chowdhury, *RiceLeafBD*, Mendeley Data, doi:10.17632/kx9rx8p2mz.1

Extract them to `external/sethy/extracted/` and `external/riceleafbd/extracted/` under the data root.
`data/class_mapping.csv` maps their labels to Paddy Doctor classes.

No images are redistributed here.

## Running

All commands are run from the data root (default `~/joiv-data`; set `JOIV_DATA` otherwise).

```bash
cd ~/joiv-data
S=/path/to/repo/scripts
python $S/02_train.py --split splits/paddy.csv --arch effb0 --seed 0 --out runs
python $S/03_export_onnx.py --run runs/effb0_s0 --split splits/paddy.csv
python $S/10_export_int8_pct.py --run runs/effb0_s0 --split splits/paddy.csv
python $S/04_verify_cam.py --run runs/effb0_s0 --split splits/paddy.csv --n 50
python $S/05_eval_onnx.py --run runs/effb0_s0 --split splits/paddy.csv
```

`scripts/reproduce.sh` runs the full sequence (12 runs, ablations, external sets, layer sensitivity, mixed
precision, calibration size, confidence analysis, tables and figures).

| Step | Script |
| --- | --- |
| Training (two stages, AdamW, label smoothing 0.1) | `02_train.py` |
| ONNX export: FP32, FP16, INT8-dyn, QDQ MinMax 7-bit and 8-bit | `03_export_onnx.py` |
| QDQ percentile (99.999) | `10_export_int8_pct.py` |
| CAM equivalence check | `04_verify_cam.py` |
| Accuracy, flips, McNemar, CAM stability, deletion/insertion | `05_eval_onnx.py` |
| QDQ settings ablation | `07_qdq_ablation.py` |
| Map-swap insertion control | `14_insertion_check.py` |
| External datasets | `15_cross_dataset.py`, `22_external_duplicates.py` |
| Logits and CAM dumps for later analyses | `16_dump_outputs.py` |
| Layer sensitivity and mixed precision | `17_layer_quant_error.py`, `18_mixed_precision.py` |
| Confidence-based rejection | `19_ood_guard.py` |
| Per-class analysis | `20_per_class.py` |
| Calibration set size | `21_calib_size.py` |
| Phone–PC agreement | `09_compare_phone.py` |
| Tables and figures | `11`, `12`, `13`, `23`, `24`, `25` |

## Android benchmark

See `android/README.md`. `08_android_assets.py` copies the seed-0 models and the 100 benchmark images
(`data/android_bench_index.csv`) into the app before building it.

## Citation

See `CITATION.cff`.

## License

Code: MIT (`LICENSE`). The datasets keep their own licenses.
