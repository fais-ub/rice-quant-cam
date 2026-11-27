#!/usr/bin/env bash
# Run from the data root (contains splits/, raw/, external/, phone/). Takes about two days on one GPU workstation.
set -euo pipefail
S=$(cd "$(dirname "$0")" && pwd)
export JOIV_DATA=${JOIV_DATA:-$PWD}
SPLIT=splits/paddy.csv
RUNS=$(echo runs/{mnv2,mnv3l,effl0,effb0}_s{0,1,2})
PC=results/paper_pc
DUMPS=$PC/dumps
CROSS=$PC/cross_dataset
MP=$PC/mixed_precision
LE=$MP/layer_error
CAL=$PC/calib_size
MP_LEVELS="pct se stem stem_se top1 top2 top4 top8 top16"

bash "$S/run_matrix.sh" $SPLIT
for R in mnv2_s0 mnv3l_s0 effb0_s0; do python "$S/07_qdq_ablation.py" --run runs/$R --split $SPLIT; done
for R in $RUNS; do python "$S/14_insertion_check.py" --run $R --split $SPLIT; done

python "$S/15_cross_dataset.py" prepare --dataset sethy --root "external/sethy/extracted/Rice Leaf Disease Images" \
  --mapping "$S/../data/class_mapping.csv" --out $CROSS
python "$S/15_cross_dataset.py" prepare --dataset riceleafbd --root "external/riceleafbd/extracted/Original Images" \
  --mapping "$S/../data/class_mapping.csv" --out $CROSS
for D in sethy riceleafbd; do python "$S/15_cross_dataset.py" eval --dataset $D --runs $RUNS --out $CROSS; done
python "$S/15_cross_dataset.py" summarize --out $CROSS --data .

python "$S/16_dump_outputs.py" prep --split $SPLIT --out $DUMPS
for R in $RUNS; do
  python "$S/16_dump_outputs.py" dump --run $R --variants fp32 int8_dyn int8_qdq int8_qdq_fr int8_qdq_pct --out $DUMPS
done

for R in effb0_s0 mnv3l_s0 mnv2_s0 effl0_s0; do
  python "$S/17_layer_quant_error.py" --run runs/$R --split $SPLIT --out $LE --n 128
done
python "$S/17_layer_quant_error.py" summarize --out $LE

for R in runs/{effb0,mnv3l}_s{0,1,2}; do
  python "$S/18_mixed_precision.py" build --run $R --split $SPLIT --out $MP --rank $LE --levels $MP_LEVELS
  V=""
  for L in $MP_LEVELS; do V="$V mp_$L=$MP/models/$(basename $R)/mp_$L.onnx"; done
  python "$S/16_dump_outputs.py" dump --run $R --out $DUMPS --variants fp32 $V
done
python "$S/18_mixed_precision.py" latency --runs runs/effb0_s0 runs/mnv3l_s0 --out $MP
python "$S/18_mixed_precision.py" summarize --runs runs/{effb0,mnv3l}_s{0,1,2} --dumps $DUMPS --out $MP
python "$S/18_mixed_precision.py" figure --out $MP

for R in $RUNS; do
  python "$S/21_calib_size.py" build --run $R --split $SPLIT --out $CAL
  V=""
  for M in minmax percentile; do for N in 32 100 300 1000; do
    V="$V calib_${M}_n$N=$CAL/models/$(basename $R)/${M}_n$N.onnx"
  done; done
  python "$S/16_dump_outputs.py" dump --run $R --datasets paddy_test --out $DUMPS --variants fp32 $V
done
python "$S/21_calib_size.py" summarize --dumps $DUMPS --out results/paper_final/_regen/calib_size
python "$S/21_calib_size.py" figure --out results/paper_final/_regen/calib_size

python "$S/19_ood_guard.py" --dumps $DUMPS --out $PC/ood_guard
python "$S/20_per_class.py" --data . --out $PC/per_class
python "$S/22_external_duplicates.py" --data . --out $CROSS/duplicates
python "$S/13_qualitative_cams.py" compute --runs runs/effb0_s0 runs/mnv2_s0 --split $SPLIT --out $PC/cams_qualitative.npz
python "$S/13_qualitative_cams.py" plot --npz $PC/cams_qualitative.npz --out $PC

python "$S/11_phone_tables.py" --phone-root phone --out results/paper_phone \
  --dirs files/bench_Xiaomi_2306EPN60G_1795484469 files/bench_Xiaomi_2306EPN60G_1795484883 \
         files/bench_Xiaomi_2306EPN60G_1795485167 files/bench_Xiaomi_2306EPN60G_1795485421 \
         'motorola_g45/bench_*' 'redmi_note11pro5g/bench_*' 'oppo_cph2687/bench_*'
python "$S/12_pc_tables.py" --data . --out $PC \
  --runvar-dirs phone/files/bench_Xiaomi_2306EPN60G_1795483073 phone/files/bench_Xiaomi_2306EPN60G_1795483721 \
                phone/files/bench_Xiaomi_2306EPN60G_1795485167
python "$S/23_paper_final.py"
python "$S/24_joiv_figures.py"
python "$S/25_print_figures.py"
