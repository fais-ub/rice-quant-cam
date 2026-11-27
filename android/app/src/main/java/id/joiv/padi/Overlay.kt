package id.joiv.padi

import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.max
import kotlin.math.min

object Overlay {
    fun upsample(cam: FloatArray, h: Int, w: Int, size: Int): FloatArray {
        val out = FloatArray(size * size)
        for (y in 0 until size) for (x in 0 until size) {
            val fy = ((y + 0.5f) * h / size - 0.5f).coerceIn(0f, (h - 1).toFloat())
            val fx = ((x + 0.5f) * w / size - 0.5f).coerceIn(0f, (w - 1).toFloat())
            val y0 = fy.toInt(); val x0 = fx.toInt(); val y1 = min(y0 + 1, h - 1); val x1 = min(x0 + 1, w - 1)
            val dy = fy - y0; val dx = fx - x0
            out[y * size + x] = cam[y0 * w + x0] * (1 - dy) * (1 - dx) + cam[y0 * w + x1] * (1 - dy) * dx +
                cam[y1 * w + x0] * dy * (1 - dx) + cam[y1 * w + x1] * dy * dx
        }
        val lo = out.min(); val hi = out.max(); val d = max(hi - lo, 1e-8f)
        for (i in out.indices) out[i] = (out[i] - lo) / d
        return out
    }

    private fun jet(v: Float): Int {
        val r = (1.5f - kotlin.math.abs(4f * v - 3f)).coerceIn(0f, 1f)
        val g = (1.5f - kotlin.math.abs(4f * v - 2f)).coerceIn(0f, 1f)
        val b = (1.5f - kotlin.math.abs(4f * v - 1f)).coerceIn(0f, 1f)
        return Color.rgb((r * 255).toInt(), (g * 255).toInt(), (b * 255).toInt())
    }

    fun blend(rgb: IntArray, heat: FloatArray, size: Int, alpha: Float = 0.45f): Bitmap {
        val px = IntArray(size * size)
        for (i in px.indices) {
            val j = jet(heat[i])
            val r = ((1 - alpha) * rgb[i * 3] + alpha * Color.red(j)).toInt()
            val g = ((1 - alpha) * rgb[i * 3 + 1] + alpha * Color.green(j)).toInt()
            val b = ((1 - alpha) * rgb[i * 3 + 2] + alpha * Color.blue(j)).toInt()
            px[i] = Color.rgb(r, g, b)
        }
        return Bitmap.createBitmap(px, size, size, Bitmap.Config.ARGB_8888)
    }
}
