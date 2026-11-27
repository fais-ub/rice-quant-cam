package id.joiv.padi

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContent { MaterialTheme { App() } }
    }

    private fun decode(uri: Uri): Bitmap =
        ImageDecoder.decodeBitmap(ImageDecoder.createSource(contentResolver, uri)) { d, _, _ ->
            d.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
        }.copy(Bitmap.Config.ARGB_8888, false)

    @OptIn(ExperimentalLayoutApi::class)
    @Composable
    fun App() {
        val bench = remember { Benchmark(this) { } }
        val dirs = remember { bench.modelDirs() }
        var dir by remember { mutableStateOf(dirs.firstOrNull() ?: "") }
        var precision by remember { mutableStateOf("int8_qdq_pct") }
        var overlay by remember { mutableStateOf<Bitmap?>(null) }
        var text by remember { mutableStateOf(if (dirs.isEmpty()) "No models in assets/models/" else "Select a rice leaf photo") }
        val logLines = remember { mutableStateListOf<String>() }
        var busy by remember { mutableStateOf(false) }
        val scope = rememberCoroutineScope()

        fun classify(bmp: Bitmap) = scope.launch {
            busy = true
            val (img, msg) = withContext(Dispatchers.Default) {
                val head = Head(assets.open("models/$dir/head.json").bufferedReader().readText())
                val rgb = Preprocess.resizeLikePil(bmp)
                Model(this@MainActivity, "models/$dir", precision, Ep.XNNPACK, 4).use { m ->
                    val x = Preprocess.toTensor(rgb, head.mean, head.std)
                    m.run(x)
                    val t = SystemClock.elapsedRealtimeNanos(); val r = m.run(x)
                    val ms = (SystemClock.elapsedRealtimeNanos() - t) / 1e6
                    val top3 = r.probs.indices.sortedByDescending { r.probs[it] }.take(3)
                    val cam = Cam.compute(r, head, r.top)
                    val heat = Overlay.upsample(cam, r.h, r.wd, Preprocess.SIZE)
                    Overlay.blend(rgb, heat, Preprocess.SIZE) to
                        top3.joinToString("\n") { "${head.classes[it]}: ${"%.1f".format(r.probs[it] * 100)}%" } +
                        "\n($dir, $precision, ${"%.1f".format(ms)} ms)"
                }
            }
            overlay = img; text = msg; busy = false
        }

        val pick = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
            if (uri != null) classify(decode(uri))
        }

        Column(Modifier.fillMaxSize().padding(16.dp).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text("Rice XAI", style = MaterialTheme.typography.headlineSmall)
            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                dirs.forEach { d -> FilterChip(selected = d == dir, onClick = { dir = d }, label = { Text(d) }) }
            }
            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                bench.precisions.filter { assets.list("models/$dir")?.contains("$it.onnx") == true }.forEach { p -> FilterChip(selected = p == precision, onClick = { precision = p }, label = { Text(p) }) }
            }
            Button(enabled = !busy && dirs.isNotEmpty(), onClick = {
                pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
            }) { Text("Select photo") }
            overlay?.let { Image(it.asImageBitmap(), null, Modifier.fillMaxWidth().aspectRatio(1f)) }
            Text(text)
            HorizontalDivider()
            OutlinedButton(enabled = !busy && dirs.isNotEmpty(), onClick = {
                busy = true; logLines.clear()
                scope.launch(Dispatchers.Default) {
                    Benchmark(this@MainActivity) { line -> scope.launch { logLines.add(0, line) } }.run(dir)
                    busy = false
                }
            }) { Text("Run benchmark ($dir)") }
            OutlinedButton(enabled = !busy && dirs.isNotEmpty(), onClick = {
                busy = true; logLines.clear()
                scope.launch(Dispatchers.Default) {
                    Benchmark(this@MainActivity) { line -> scope.launch { logLines.add(0, line) } }.runAll()
                    busy = false
                }
            }) { Text("Run all models (2 min cool-down)") }
            Button(enabled = !busy, onClick = {
                busy = true
                scope.launch {
                    val zip = withContext(Dispatchers.IO) { runCatching { Benchmark(this@MainActivity) { }.zipResults() } }
                    busy = false
                    val f = zip.getOrNull()
                    if (f == null) {
                        logLines.add(0, zip.exceptionOrNull()?.let { "Could not create ZIP: ${it.message}" } ?: "No benchmark results yet")
                    } else {
                        logLines.add(0, "ZIP ready: ${f.name} (${"%.1f".format(f.length() / 1e6)} MB)")
                        val uri = FileProvider.getUriForFile(this@MainActivity, "$packageName.fileprovider", f)
                        val send = Intent(Intent.ACTION_SEND).setType("application/zip")
                            .putExtra(Intent.EXTRA_STREAM, uri).putExtra(Intent.EXTRA_SUBJECT, f.nameWithoutExtension)
                            .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                        startActivity(Intent.createChooser(send, "Share benchmark results"))
                    }
                }
            }) { Text("Share results (ZIP)") }
            logLines.take(30).forEach { Text(it, style = MaterialTheme.typography.bodySmall) }
        }
    }
}
