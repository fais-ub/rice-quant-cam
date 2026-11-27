# Android benchmark app

Kotlin, Jetpack Compose and ONNX Runtime Android 1.32.0 (CPU and XNNPACK execution providers), arm64-v8a only.

| File | Role |
| --- | --- |
| `Preprocess.kt` | PIL-compatible antialiased bilinear resize to 224 × 224, then normalization from `head.json` |
| `Model.kt` | Loads `<precision>.onnx`, returns `logits` and `features`; `Cam.compute` builds the CAM |
| `Overlay.kt` | Heatmap overlay for the interactive view |
| `Benchmark.kt` | Latency, PSS, battery temperature and thermal status; predictions and 7 × 7 CAMs on the bench images |
| `MainActivity.kt` | Photo classification, benchmark buttons, result sharing as ZIP |

Each configuration (precision × provider × 1/2/4 threads) gets 20 warm-up calls and 200 timed calls on one fixed
input. FP16 runs on the CPU provider only because XNNPACK rejects its FP16 convolutions. "Run all models" waits
120 s between models.

## Build and run

1. Copy models and bench images (seed 0, from the data root):

   ```bash
   python scripts/08_android_assets.py --runs runs/mnv2_s0 runs/mnv3l_s0 runs/effl0_s0 runs/effb0_s0 \
       --split splits/paddy.csv --n 100
   ```

   For the mixed-precision rows, also copy `results/paper_pc/mixed_precision/models/<run>/mp_top4.onnx` to
   `app/src/main/assets/models/<run>/int8_qdq_mp.onnx` for `effb0_s0` and `mnv3l_s0`.

2. Build and install:

   ```bash
   ./gradlew assembleRelease
   adb install -r app/build/outputs/apk/release/app-release.apk
   ```

3. Measure with the phone charging, in airplane mode, screen on, and RAM extension features off.

4. Collect results with `adb pull /sdcard/Android/data/id.joiv.padi/files/ phone/` or the share button, then compare
   with the PC:

   ```bash
   python scripts/09_compare_phone.py --bench phone/<folder>/predictions.csv --runs-root runs \
       --assets android/app/src/main/assets
   ```
