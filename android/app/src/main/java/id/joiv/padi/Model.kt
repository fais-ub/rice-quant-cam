package id.joiv.padi

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.json.JSONObject
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.max

class Head(json: String) {
    private val o = JSONObject(json)
    val arch: String = o.getString("arch")
    val classes: List<String> = o.getJSONArray("classes").let { a -> List(a.length()) { a.getString(it) } }
    val mean = o.getJSONArray("mean").let { a -> FloatArray(3) { a.getDouble(it).toFloat() } }
    val std = o.getJSONArray("std").let { a -> FloatArray(3) { a.getDouble(it).toFloat() } }
    val w: Array<FloatArray> = o.getJSONArray("fc_weight").let { a ->
        Array(a.length()) { r -> a.getJSONArray(r).let { row -> FloatArray(row.length()) { row.getDouble(it).toFloat() } } }
    }
}

class Result(val logits: FloatArray, val features: FloatArray, val c: Int, val h: Int, val wd: Int) {
    val probs: FloatArray by lazy {
        val m = logits.max(); val e = logits.map { exp((it - m).toDouble()) }; val s = e.sum()
        FloatArray(logits.size) { (e[it] / s).toFloat() }
    }
    val top: Int get() = logits.indices.maxBy { logits[it] }
}

enum class Ep { CPU, XNNPACK }

class Model(ctx: Context, val dir: String, val precision: String, ep: Ep, threads: Int) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    init {
        val bytes = ctx.assets.open("$dir/$precision.onnx").use { it.readBytes() }
        val so = OrtSession.SessionOptions()
        when (ep) {
            Ep.CPU -> so.setIntraOpNumThreads(threads)
            Ep.XNNPACK -> {
                so.setIntraOpNumThreads(1)
                so.addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
            }
        }
        so.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        session = env.createSession(bytes, so)
    }

    fun run(x: FloatArray): Result {
        OnnxTensor.createTensor(env, FloatBuffer.wrap(x), longArrayOf(1, 3, Preprocess.SIZE.toLong(), Preprocess.SIZE.toLong())).use { t ->
            session.run(mapOf("image" to t)).use { out ->
                val logits = (out.get("logits").get() as OnnxTensor).floatBuffer.let { b -> FloatArray(b.remaining()).also { b.get(it) } }
                val ft = out.get("features").get() as OnnxTensor
                val shape = ft.info.shape
                val feats = ft.floatBuffer.let { b -> FloatArray(b.remaining()).also { b.get(it) } }
                return Result(logits, feats, shape[1].toInt(), shape[2].toInt(), shape[3].toInt())
            }
        }
    }

    override fun close() = session.close()
}

object Cam {
    fun compute(r: Result, head: Head, cls: Int): FloatArray {
        val hw = r.h * r.wd
        val out = FloatArray(hw)
        val w = head.w[cls]
        for (k in 0 until r.c) { val wk = w[k]; val base = k * hw; for (p in 0 until hw) out[p] += wk * r.features[base + p] }
        for (p in 0 until hw) out[p] = max(0f, out[p]) / hw
        return out
    }
}
