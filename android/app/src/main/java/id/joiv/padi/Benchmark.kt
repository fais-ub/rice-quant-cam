package id.joiv.padi

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.BitmapFactory
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import android.os.SystemClock
import java.io.File
import java.util.Locale


class Benchmark(private val ctx: Context, private val log: (String) -> Unit) {
    val warmup = 20
    val runs = 200
    val threadsList = listOf(1, 2, 4)
    val precisions = listOf("fp32", "fp16", "int8_dyn", "int8_qdq", "int8_qdq_fr", "int8_qdq_pct", "int8_qdq_mp")

    private fun pssKb(): Int = Debug.MemoryInfo().also { Debug.getMemoryInfo(it) }.totalPss

    fun batteryTempC(): Float =
        (ctx.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            ?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -10) ?: -10) / 10f

    fun thermalStatus(): Int =
        if (Build.VERSION.SDK_INT >= 29) (ctx.getSystemService(Context.POWER_SERVICE) as PowerManager).currentThermalStatus else -1

    fun modelDirs(): List<String> = ctx.assets.list("models")?.filter { ctx.assets.list("models/$it")?.contains("head.json") == true }?.sorted() ?: emptyList()

    fun run(onlyDir: String? = null): File {
        val device = "${Build.MANUFACTURER}_${Build.MODEL}".replace(Regex("[^A-Za-z0-9_]"), "")
        val outDir = File(ctx.getExternalFilesDir(null), "bench_${device}_${System.currentTimeMillis() / 1000}").apply { mkdirs() }
        val lat = File(outDir, "latency.csv").bufferedWriter()
        val pred = File(outDir, "predictions.csv").bufferedWriter()
        lat.write("device,soc,model,precision,ep,threads,load_ms,median_ms,p90_ms,mean_ms,min_ms,pss_kb,model_mb,temp_start_c,temp_end_c,thermal_status\n")
        pred.write("device,model,precision,ep,image,pred,prob,cam\n")
        val benchImgs = ctx.assets.list("bench")?.sorted() ?: emptyList()
        val soc = if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.HARDWARE

        for (dir in modelDirs().filter { onlyDir == null || it == onlyDir }) {
            val head = Head(ctx.assets.open("models/$dir/head.json").bufferedReader().readText())
            val present = ctx.assets.list("models/$dir")!!.toSet()
            val inputs = benchImgs.map { name ->
                val bmp = ctx.assets.open("bench/$name").use { BitmapFactory.decodeStream(it) }
                name to Preprocess.toTensor(Preprocess.resizeLikePil(bmp), head.mean, head.std)
            }
            val fixed = inputs.firstOrNull()?.second ?: FloatArray(3 * 224 * 224)
            for (p in precisions.filter { "$it.onnx" in present }) {
                val sizeMb = ctx.assets.openFd("models/$dir/$p.onnx").use { it.length / 1e6 }
                for (ep in Ep.values()) for (th in threadsList) {
                    if (p == "fp16" && ep == Ep.XNNPACK) continue
                    var tempStart = Float.NaN
                    try {
                        System.gc(); SystemClock.sleep(500)
                        tempStart = batteryTempC()
                        val t0 = SystemClock.elapsedRealtimeNanos()
                        Model(ctx, "models/$dir", p, ep, th).use { m ->
                            val loadMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
                            repeat(warmup) { m.run(fixed) }
                            val pss = pssKb()
                            val ts = DoubleArray(runs) {
                                val s = SystemClock.elapsedRealtimeNanos(); m.run(fixed)
                                (SystemClock.elapsedRealtimeNanos() - s) / 1e6
                            }.sorted()
                            val tempEnd = batteryTempC()
                            val thermal = thermalStatus()
                            lat.write("$device,$soc,$dir,$p,$ep,$th,${"%.1f".format(Locale.US, loadMs)},${"%.2f".format(Locale.US, ts[runs / 2])}," +
                                "${"%.2f".format(Locale.US, ts[(runs * 0.9).toInt()])},${"%.2f".format(Locale.US, ts.average())},${"%.2f".format(Locale.US, ts[0])},$pss,${"%.2f".format(Locale.US, sizeMb)}," +
                                "${"%.1f".format(Locale.US, tempStart)},${"%.1f".format(Locale.US, tempEnd)},$thermal\n")
                            lat.flush()
                            log("$dir $p $ep t=$th: median ${"%.1f".format(Locale.US, ts[runs / 2])} ms")
                            if (th == threadsList.last()) for ((name, x) in inputs) {
                                val r = m.run(x); val c = r.top
                                val cam = Cam.compute(r, head, c).joinToString(" ") { "%.6g".format(Locale.US, it) }
                                pred.write("$device,$dir,$p,$ep,$name,${head.classes[c]},${"%.5f".format(Locale.US, r.probs[c])},$cam\n")
                            }
                            pred.flush()
                        }
                    } catch (e: Throwable) {
                        lat.write("$device,$soc,$dir,$p,$ep,$th,ERROR,${e.javaClass.simpleName}: ${e.message?.replace(',', ';')?.take(150)},,,,,,${"%.1f".format(Locale.US, tempStart)},${"%.1f".format(Locale.US, batteryTempC())},${thermalStatus()}\n")
                        log("$dir $p $ep t=$th: ERROR ${e.message}")
                    }
                }
            }
        }
        lat.close(); pred.close()
        log("Done: ${outDir.absolutePath}")
        return outDir
    }

    fun zipResults(): File? {
        val root = ctx.getExternalFilesDir(null) ?: return null
        val benches = root.listFiles { f -> f.isDirectory && f.name.startsWith("bench_") }?.sortedBy { it.name } ?: emptyList()
        if (benches.isEmpty()) return null
        val device = "${Build.MANUFACTURER}_${Build.MODEL}".replace(Regex("[^A-Za-z0-9_]"), "")
        val shareDir = File(ctx.cacheDir, "share").apply { deleteRecursively(); mkdirs() }
        val zip = File(shareDir, "results_${device}_${System.currentTimeMillis() / 1000}.zip")
        val info = buildString {
            appendLine("manufacturer=${Build.MANUFACTURER}"); appendLine("model=${Build.MODEL}")
            appendLine("device=${Build.DEVICE}"); appendLine("hardware=${Build.HARDWARE}")
            if (Build.VERSION.SDK_INT >= 31) appendLine("soc=${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}")
            appendLine("android=${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})")
            appendLine("app=${ctx.packageManager.getPackageInfo(ctx.packageName, 0).versionName}")
            appendLine("battery_temp_c=${batteryTempC()}"); appendLine("thermal_status=${thermalStatus()}")
        }
        java.util.zip.ZipOutputStream(zip.outputStream().buffered()).use { z ->
            z.putNextEntry(java.util.zip.ZipEntry("device_info.txt")); z.write(info.toByteArray()); z.closeEntry()
            for (b in benches) b.walkTopDown().filter { it.isFile }.forEach { f ->
                z.putNextEntry(java.util.zip.ZipEntry(f.relativeTo(root).invariantSeparatorsPath))
                f.inputStream().use { it.copyTo(z) }; z.closeEntry()
            }
        }
        return zip
    }

    fun runAll(cooldownMs: Long = 120_000): List<File> {
        val dirs = modelDirs()
        val outs = mutableListOf<File>()
        dirs.forEachIndexed { i, d ->
            if (i > 0) {
                log("Cooling down ${cooldownMs / 1000} s (battery ${"%.1f".format(Locale.US, batteryTempC())} °C)")
                SystemClock.sleep(cooldownMs)
            }
            outs += run(d)
        }
        log("All models done")
        return outs
    }
}
