package id.joiv.padi

import android.graphics.Bitmap
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

// Reimplements PIL's antialiased bilinear resize so phone and PC inputs match.
object Preprocess {
    const val SIZE = 224

    private class Coeffs(val start: IntArray, val len: IntArray, val w: Array<FloatArray>)

    private fun coeffs(inSize: Int, outSize: Int): Coeffs {
        val scale = inSize.toDouble() / outSize
        val fs = max(scale, 1.0)
        val support = fs
        val start = IntArray(outSize); val len = IntArray(outSize)
        val w = Array(outSize) { FloatArray(0) }
        for (i in 0 until outSize) {
            val center = (i + 0.5) * scale
            val xmin = max((center - support + 0.5).toInt(), 0)
            val xmax = min((center + support + 0.5).toInt(), inSize)
            val ww = DoubleArray(xmax - xmin)
            var sum = 0.0
            for (x in xmin until xmax) {
                val t = abs((x - center + 0.5) / fs)
                val v = if (t < 1.0) 1.0 - t else 0.0
                ww[x - xmin] = v; sum += v
            }
            start[i] = xmin; len[i] = xmax - xmin
            w[i] = FloatArray(ww.size) { (if (sum != 0.0) ww[it] / sum else 0.0).toFloat() }
        }
        return Coeffs(start, len, w)
    }

    private fun clip8(v: Float) = min(255, max(0, v.roundToInt()))

    fun resizeLikePil(bmp: Bitmap): IntArray {
        val iw = bmp.width; val ih = bmp.height
        val px = IntArray(iw * ih); bmp.getPixels(px, 0, iw, 0, 0, iw, ih)
        val cx = coeffs(iw, SIZE); val cy = coeffs(ih, SIZE)
        val tmp = IntArray(ih * SIZE * 3)
        for (y in 0 until ih) for (i in 0 until SIZE) {
            var r = 0f; var g = 0f; var b = 0f
            val s = cx.start[i]; val wt = cx.w[i]
            for (k in 0 until cx.len[i]) {
                val p = px[y * iw + s + k]; val c = wt[k]
                r += ((p shr 16) and 0xFF) * c; g += ((p shr 8) and 0xFF) * c; b += (p and 0xFF) * c
            }
            val o = (y * SIZE + i) * 3
            tmp[o] = clip8(r); tmp[o + 1] = clip8(g); tmp[o + 2] = clip8(b)
        }
        val out = IntArray(SIZE * SIZE * 3)
        for (j in 0 until SIZE) {
            val s = cy.start[j]; val wt = cy.w[j]
            for (i in 0 until SIZE) for (ch in 0 until 3) {
                var v = 0f
                for (k in 0 until cy.len[j]) v += tmp[((s + k) * SIZE + i) * 3 + ch] * wt[k]
                out[(j * SIZE + i) * 3 + ch] = clip8(v)
            }
        }
        return out
    }

    fun toTensor(rgb: IntArray, mean: FloatArray, std: FloatArray): FloatArray {
        val n = SIZE * SIZE
        val x = FloatArray(3 * n)
        for (p in 0 until n) for (c in 0 until 3) x[c * n + p] = (rgb[p * 3 + c] / 255f - mean[c]) / std[c]
        return x
    }
}
