"""Why INT8 breaks the SE/swish models: per-layer quantisation error, activation ranges and single-tensor sensitivity."""
import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

QMODELS = {"minmax_rr": "int8_qdq", "minmax": "int8_qdq_fr", "pct": "int8_qdq_pct"}
MODELS = {"mnv2": "MobileNetV2", "mnv3l": "MobileNetV3-L", "effl0": "EfficientNet-Lite0", "effb0": "EfficientNet-B0"}


def fpname(t):
    """Tensor name in the FP32 model (the quantiser renames a graph output's producer tensor)."""
    return t[: -len("_QuantizeLinear_Input")] if t.endswith("_QuantizeLinear_Input") else t


def act_quant_map(qm):
    """Activation tensors quantised in a QDQ model: T -> (scale, zero_point, qmin, qmax, dq_output)."""
    import onnx
    from onnx import numpy_helper

    inits = {i.name: i for i in qm.graph.initializer}
    out = {}
    for n in qm.graph.node:
        if n.op_type != "QuantizeLinear" or n.input[0] in inits:
            continue
        s = numpy_helper.to_array(inits[n.input[1]]).astype(np.float32)
        zp = numpy_helper.to_array(inits[n.input[2]])
        info = np.iinfo(zp.dtype)
        dq = [m.output[0] for m in qm.graph.node if m.op_type == "DequantizeLinear" and m.input[0] == n.output[0]]
        out[n.input[0]] = dict(scale=float(s), zp=int(zp), qmin=info.min, qmax=info.max, dq=dq[0], dtype=zp.dtype,
                               q_node=n.name, scale_name=n.input[1], zp_name=n.input[2],
                               scale_init=copy.deepcopy(inits[n.input[1]]), zp_init=copy.deepcopy(inits[n.input[2]]))
    del onnx
    return out


def describe(tensor, producers):
    n = producers.get(tensor)
    name = n.name if n is not None and n.name.startswith("/") else tensor
    m = re.search(r"blocks\.(\d+)/blocks\.\d+\.(\d+)", name)
    block = f"{m.group(1)}.{m.group(2)}" if m else ("stem" if ("conv_stem" in name or "bn1" in name.split("/")[1:2] or tensor == "image")
                                                    else "head")
    if "/se/" in name:
        role = "se_" + ("gate" if re.search(r"/gate/", name) else "mul" if name.endswith("/se/Mul") else
                        "squeeze" if "ReduceMean" in name else "fc" if "Conv" in name else "act")
    elif "conv_dw" in name:
        role = "dw_conv"
    elif "conv_stem" in name:
        role = "stem_conv"
    elif "conv_pwl" in name:
        role = "pwl_conv"
    elif "conv_pw" in name or "conv_head" in name:
        role = "pw_conv"
    elif n is not None and n.op_type in ("Sigmoid", "HardSigmoid", "Mul", "Relu", "Clip", "HardSwish"):
        role = "act"
    elif n is not None and n.op_type == "Add":
        role = "residual_add"
    else:
        role = (n.op_type.lower() if n is not None else "input")
    return {"producer": name, "op": n.op_type if n is not None else "Input", "block": block, "role": role}


def add_outputs(model, names):
    from onnx import TensorProto, helper

    m = copy.deepcopy(model)
    have = {o.name for o in m.graph.output}
    for t in names:
        if t not in have:
            m.graph.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
    return m


def session(model, threads, optimise=False):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.log_severity_level = 3
    if not optimise:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])


def insert_fake_quant(fp, qmap, tensors):
    """FP32 model with Q->DQ (scale/zp from qmap) after each tensor in `tensors` (consumers rewired)."""
    from onnx import helper

    m = copy.deepcopy(fp)
    have_init = {i.name for i in m.graph.initializer}
    new_nodes = []
    for tqname in tensors:
        q = qmap[tqname]
        t = fpname(tqname)
        for init in (q["scale_init"], q["zp_init"]):
            if init.name not in have_init:
                m.graph.initializer.append(init)
                have_init.add(init.name)
        tq, tdq = f"{t}__fq_q", f"{t}__fq_dq"
        for n in m.graph.node:
            for k, inp in enumerate(n.input):
                if inp == t:
                    n.input[k] = tdq
        new_nodes += [helper.make_node("QuantizeLinear", [t, q["scale_name"], q["zp_name"]], [tq], name=f"{t}__fqQ"),
                      helper.make_node("DequantizeLinear", [tq, q["scale_name"], q["zp_name"]], [tdq], name=f"{t}__fqDQ")]
    m.graph.node.extend(new_nodes)
    return toposort(m)


def strip_act_quant(qm, qmap, only=None, fp=None):
    """Remove activation Q->DQ pairs from a QDQ model: all of them (weights-only model) or those in `only`."""
    from onnx import helper

    m = copy.deepcopy(qm)
    qmap = {t: q for t, q in qmap.items() if only is None or t in only}
    rewire = {q["dq"]: t for t, q in qmap.items()}
    qprod = {o: n.op_type for n in m.graph.node for o in n.output}
    fp_producers = {o: n for n in fp.graph.node for o in n.output} if fp is not None else {}
    fp_inits = {i.name: i for i in fp.graph.initializer} if fp is not None else {}
    have = {i.name for i in m.graph.initializer}
    extra = []
    for dq, t in list(rewire.items()):
        fpn = fp_producers.get(fpname(t))
        if fpn is not None and fpn.op_type in ("Relu", "Clip") and qprod.get(t) != fpn.op_type:
            act = copy.deepcopy(fpn)
            act.input[0] = t
            del act.output[:]
            act.output.append(f"{t}__act")
            act.name = f"{t}__act"
            for i in act.input[1:]:
                if i and i not in have and i in fp_inits:
                    m.graph.initializer.append(copy.deepcopy(fp_inits[i]))
                    have.add(i)
            extra.append(act)
            rewire[dq] = f"{t}__act"
    outs = {o.name for o in m.graph.output}
    drop = set()
    for n in m.graph.node:
        if n.op_type == "QuantizeLinear" and n.input[0] in qmap:
            drop.add(n.name)
        if n.op_type == "DequantizeLinear" and n.output[0] in rewire:
            drop.add(n.name)
    keep = [n for n in m.graph.node if n.name not in drop]
    keep += extra
    for n in keep:
        if n in extra:
            continue
        for k, inp in enumerate(n.input):
            if inp in rewire:
                n.input[k] = rewire[inp]
    keep += [helper.make_node("Identity", [rewire[o]], [o], name=f"{o}__id") for o in outs if o in rewire]
    del m.graph.node[:]
    m.graph.node.extend(keep)
    return toposort(m)


def toposort(m):
    avail = {i.name for i in m.graph.input} | {i.name for i in m.graph.initializer} | {""}
    nodes, out = list(m.graph.node), []
    while nodes:
        rest = []
        for n in nodes:
            if all(i in avail for i in n.input):
                out.append(n)
                avail.update(n.output)
            else:
                rest.append(n)
        if len(rest) == len(nodes):
            raise RuntimeError("graph has a cycle or a missing input")
        nodes = rest
    del m.graph.node[:]
    m.graph.node.extend(out)
    return m


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def logits_of(sess, xs):
    return np.concatenate([sess.run(["logits"], {"image": x})[0] for x in xs])


def kl(p, q):
    return float(np.mean(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)), 1)))


def analyse(a):
    import onnx
    from onnxruntime.quantization.shape_inference import quant_pre_process

    from joiv.data import load_split, preprocess_numpy

    run = Path(a.run)
    out = Path(a.out) / run.name
    out.mkdir(parents=True, exist_ok=True)
    head = json.loads((run / "onnx" / "head.json").read_text())
    classes = head["classes"]
    pre_path = out / "fp32_pre.onnx"
    if not pre_path.exists():
        quant_pre_process(str(run / "onnx" / "fp32.onnx"), str(pre_path))
    fp = onnx.load(str(pre_path))
    producers = {o: n for n in fp.graph.node for o in n.output}
    order = {o: i for i, n in enumerate(fp.graph.node) for o in n.output}
    order["image"] = -1

    va = load_split(a.split, "val").sample(n=a.n, random_state=0).reset_index(drop=True)
    xs = [preprocess_numpy(p, head["mean"], head["std"]) for p in va.path]
    y = np.array([classes.index(l) for l in va.label])

    t0 = time.time()
    rows, sens = [], []
    fp_sess_plain = session(fp, a.threads)
    p_ref = softmax(logits_of(fp_sess_plain, xs))
    ref_top = p_ref.argmax(1)
    sens.append({"qmodel": "none", "variant": "fp32", "tensor": "", "kl": 0.0, "flip": 0.0, "acc": float(np.mean(ref_top == y))})
    for tag, fname in QMODELS.items():
        qpath = run / "onnx" / f"{fname}.onnx"
        if not qpath.exists():
            continue
        qm = onnx.load(str(qpath))
        qmap = act_quant_map(qm)
        tensors = sorted(qmap, key=lambda t: order.get(fpname(t), 10 ** 6))
        qprod = {o: n.op_type for n in qm.graph.node for o in n.output}
        fp_inits = {i.name: onnx.numpy_helper.to_array(i) for i in fp.graph.initializer}
        folded = {}
        for t in tensors:
            fpn = producers.get(fpname(t))
            if fpn is not None and fpn.op_type in ("Relu", "Clip") and qprod.get(t) != fpn.op_type:
                lo = float(fp_inits[fpn.input[1]]) if fpn.op_type == "Clip" and len(fpn.input) > 1 and fpn.input[1] else 0.0
                hi = float(fp_inits[fpn.input[2]]) if fpn.op_type == "Clip" and len(fpn.input) > 2 and fpn.input[2] else np.inf
                folded[t] = (lo, hi)
        fp_sess = session(add_outputs(fp, [fpname(t) for t in tensors if t != "image"]), a.threads)
        q_sess = session(add_outputs(qm, tensors + [qmap[t]["dq"] for t in tensors]), a.threads)
        fp_names = [o.name for o in fp_sess.get_outputs()]
        q_names = [o.name for o in q_sess.get_outputs()]
        acc = {t: dict(sig=0.0, noise_loc=0.0, sig_loc=0.0, noise_cum=0.0, clip=0, cnt=0, amax=0.0, amin=0.0,
                       absmax=0.0, p9999=[], absmean=[], steps=[]) for t in tensors}
        for x in xs:
            fo = dict(zip(fp_names, fp_sess.run(None, {"image": x})))
            fo["image"] = x
            qo = dict(zip(q_names, q_sess.run(None, {"image": x})))
            for t in tensors:
                q, r = qmap[t], acc[t]
                f32, tq, tdq = fo[fpname(t)].astype(np.float64), qo[t].astype(np.float64), qo[q["dq"]].astype(np.float64)
                if t in folded:
                    tq = np.clip(tq, *folded[t])
                lo, hi = q["scale"] * (q["qmin"] - q["zp"]), q["scale"] * (q["qmax"] - q["zp"])
                r["sig"] += float(np.sum(f32 ** 2))
                r["noise_cum"] += float(np.sum((tdq - f32) ** 2))
                r["sig_loc"] += float(np.sum(tq ** 2))
                r["noise_loc"] += float(np.sum((tdq - tq) ** 2))
                r["clip"] += int(np.sum((tq < lo) | (tq > hi)))
                r["cnt"] += tq.size
                r["amax"] = max(r["amax"], float(f32.max()))
                r["amin"] = min(r["amin"], float(f32.min()))
                ab = np.abs(f32)
                r["absmax"] = max(r["absmax"], float(ab.max()))
                r["p9999"].append(float(np.percentile(ab, 99.99)))
                r["absmean"].append(float(ab.mean()))
                lo_p, hi_p = np.percentile(tq, [0.05, 99.95])
                r["steps"].append(float((hi_p - lo_p) / q["scale"]))
        for t in tensors:
            q, r = qmap[t], acc[t]
            db = lambda s, n: float(10 * np.log10(s / n)) if n > 0 else float("inf")  # noqa: E731
            rows.append({"run": run.name, "qmodel": tag, "tensor": t, "order": order.get(fpname(t), -2),
                         **describe(fpname(t), producers),
                         "fp32_min": r["amin"], "fp32_max": r["amax"], "fp32_absmax": r["absmax"],
                         "fp32_abs_p9999": float(np.mean(r["p9999"])), "fp32_abs_mean": float(np.mean(r["absmean"])),
                         "scale": q["scale"], "zero_point": q["zp"], "act_dtype": str(np.dtype(q["dtype"])),
                         "grid_lo": q["scale"] * (q["qmin"] - q["zp"]), "grid_hi": q["scale"] * (q["qmax"] - q["zp"]),
                         "clip_frac": r["clip"] / r["cnt"], "steps_p999": float(np.mean(r["steps"])),
                         "local_sqnr_db": db(r["sig_loc"], r["noise_loc"]), "cum_sqnr_db": db(r["sig"], r["noise_cum"])})
        print(f"{run.name} {tag}: error pass {len(tensors)} tensors ({len(folded)} with folded Relu/Clip), "
              f"{time.time() - t0:.0f}s", flush=True)
        if a.error_only:
            continue

        def add_ref(variant, model, optimise=False, tensor=""):
            p = softmax(logits_of(session(model, a.threads, optimise), xs))
            sens.append({"qmodel": tag, "variant": variant, "tensor": tensor, "kl": kl(p_ref, p),
                         "flip": float(np.mean(p.argmax(1) != ref_top)), "acc": float(np.mean(p.argmax(1) == y))})

        add_ref("full_int8_kernels", qm, optimise=True)
        add_ref("full_fake_quant", qm)
        add_ref("weights_only", strip_act_quant(qm, qmap, fp=fp))
        add_ref("activations_only", insert_fake_quant(fp, qmap, tensors))
        if not a.no_sens and tag != "minmax_rr":
            for t in tensors:
                add_ref("single", insert_fake_quant(fp, qmap, [t]), tensor=t)
                add_ref("leave_one_out", strip_act_quant(qm, qmap, only={t}, fp=fp), tensor=t)
            print(f"{run.name} {tag}: sensitivity done, {time.time() - t0:.0f}s", flush=True)
    L = pd.DataFrame(rows)
    if a.error_only:
        L.to_csv(out / "layer_error.csv", index=False)
        print(L.groupby(["qmodel", "role"]).local_sqnr_db.median().round(1).to_string())
        return
    S = pd.DataFrame(sens)
    S = S.merge(L[["qmodel", "tensor", "producer", "op", "block", "role", "order"]], on=["qmodel", "tensor"], how="left")
    L.to_csv(out / "layer_error.csv", index=False)
    S.to_csv(out / "sensitivity.csv", index=False)
    print(S[S.variant != "single"].round(4).to_string(index=False))
    for tag in L.qmodel.unique():
        s = S[(S.qmodel == tag) & (S.variant == "single")].sort_values("kl", ascending=False).head(8)
        print(tag, "\n", s[["tensor", "role", "block", "kl", "flip"]].round(4).to_string(index=False))


def summarize(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(a.out)
    L = pd.concat([pd.read_csv(f) for f in sorted(out.glob("*/layer_error.csv"))], ignore_index=True)
    S = pd.concat([pd.read_csv(f).assign(run=f.parent.name) for f in sorted(out.glob("*/sensitivity.csv"))], ignore_index=True)
    L["model"] = L.run.str.split("_s").str[0]
    S["model"] = S.run.str.split("_s").str[0]
    L.to_csv(out / "layer_error_all.csv", index=False)
    S.to_csv(out / "sensitivity_all.csv", index=False)

    full = S[S.variant == "full_fake_quant"].set_index(["run", "qmodel"]).kl
    S["loo_gain"] = np.where(S.variant == "leave_one_out",
                             S.set_index(["run", "qmodel"]).index.map(full).to_numpy() - S.kl, np.nan)
    S.to_csv(out / "sensitivity_all.csv", index=False)
    loo = S[S.variant == "leave_one_out"]
    top_loo = loo.sort_values("loo_gain", ascending=False).groupby(["model", "qmodel"]).head(10)
    top_loo.to_csv(out / "top10_loo_tensors.csv", index=False)
    ss = S[S.variant == "single"]
    R = (L.groupby(["model", "qmodel", "role"]).agg(n=("tensor", "size"), steps_p999_median=("steps_p999", "median"),
                                                   local_sqnr_db_median=("local_sqnr_db", "median"),
                                                   clip_frac_mean=("clip_frac", "mean")).reset_index()
         .merge(ss.groupby(["model", "qmodel", "role"]).agg(kl_sum=("kl", "sum"), kl_max=("kl", "max"),
                                                           flip_max=("flip", "max")).reset_index(),
                on=["model", "qmodel", "role"], how="left"))
    R.to_csv(out / "role_summary.csv", index=False)
    top = ss.sort_values("kl", ascending=False).groupby(["model", "qmodel"]).head(10)
    top.to_csv(out / "top10_sensitive_tensors.csv", index=False)
    ref = S[S.variant != "single"].pivot_table(index=["model", "qmodel"], columns="variant", values="acc").reset_index()
    ref.to_csv(out / "reference_acc.csv", index=False)

    names = {"minmax_rr": "MinMax (7-bit w.)", "minmax": "MinMax", "pct": "Percentile"}
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
             r"\caption{Where static INT8 breaks (seed 0, 128 validation images, simulated QDQ). KL: KL divergence of the "
             r"fully quantised model to FP32. Gain: how much KL drops when only this activation tensor is left in FP32 "
             r"(leave-one-out). Steps: INT8 steps spanned by the central 99.9\,\% of the tensor's values (few steps = the "
             r"range is spent on rare outliers); max/p99.99: FP32 range ratio.}",
             r"\label{tab:layer_sensitivity}", r"\begin{tabular}{lllrrrr}", r"\toprule",
             r"Model (full KL) & Calib. & Tensor (block, role) & Gain & Steps & SQNR (dB) & max/p99.99 \\", r"\midrule"]
    for m in [m for m in MODELS if m in set(L.model)]:
        for tag in ["minmax", "pct"]:
            fk = S[(S.model == m) & (S.qmodel == tag) & (S.variant == "full_fake_quant")].kl
            t = loo[(loo.model == m) & (loo.qmodel == tag)].sort_values("loo_gain", ascending=False).head(3)
            for k, (_, r) in enumerate(t.iterrows()):
                le = L[(L.model == m) & (L.qmodel == tag) & (L.tensor == r.tensor)].iloc[0]
                lead = f"{MODELS[m]} ({fk.iloc[0]:.3f})" if k == 0 else ""
                lines.append(f"{lead} & {names[tag] if k == 0 else ''} & {le.block}, {le.role.replace('_', ' ')} & "
                             f"{r.loo_gain:.3f} & {le.steps_p999:.0f} & {le.local_sqnr_db:.1f} & "
                             f"{le.fp32_absmax / max(le.fp32_abs_p9999, 1e-9):.1f}" + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    (out / "tab_layer_sensitivity.tex").write_text("\n".join(lines) + "\n")

    ms = [m for m in ["effb0", "mnv3l", "mnv2", "effl0"] if m in set(L.model)]
    COL = {"minmax": "#eb6834", "pct": "#2a78d6"}
    fig, axes = plt.subplots(2, len(ms), figsize=(3.1 * len(ms), 4.2), squeeze=False, sharey="row")
    for c, m in enumerate(ms):
        for tag in ["minmax", "pct"]:
            l = L[(L.model == m) & (L.qmodel == tag)].sort_values("order").reset_index(drop=True)
            xs = np.arange(len(l))
            axes[0][c].plot(xs, l.steps_p999.clip(lower=0.5), lw=0.8, color=COL[tag], label=names[tag])
            axes[1][c].plot(xs, l.cum_sqnr_db.clip(-5, 60), lw=0.8, color=COL[tag])
            se = l.role.str.startswith("se").to_numpy()
            if tag == "minmax":
                for ax in axes[:, c]:
                    for x in xs[se]:
                        ax.axvline(x, color="#e4e3df", lw=0.5, zorder=0)
        axes[0][c].set_title(MODELS[m], fontsize=8)
        axes[0][c].set_yscale("log")
        axes[1][c].set_xlabel("activation tensor (graph order; grey = SE)", fontsize=7)
        for ax in axes[:, c]:
            ax.tick_params(labelsize=7, colors="#52514e")
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel("INT8 steps covering 99.9 %", fontsize=7.5)
    axes[1][0].set_ylabel("SQNR vs FP32 so far (dB)", fontsize=7.5)
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(out.parent / f"fig_layer_error.{ext}", dpi=300)
    pd.set_option("display.width", 250)
    print(ref.round(4).to_string(index=False))
    print(R[R.model.isin(["effb0", "mnv3l"])].round(3).to_string(index=False))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--out", required=True)
        summarize(ap.parse_args())
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128, help="validation images")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-sens", action="store_true", help="skip the single-tensor sensitivity pass")
    ap.add_argument("--error-only", action="store_true", help="only redo layer_error.csv (keep sensitivity.csv)")
    analyse(ap.parse_args())


if __name__ == "__main__":
    main()
