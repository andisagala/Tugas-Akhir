package com.example.facedetector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.facedetector.MainScreenUI.MainScreenUI
import com.google.mediapipe.tasks.vision.core.RunningMode
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine
import com.example.facedetector.OverlayView
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.util.Arrays

fun getDimensions(array: Any): List<Float> {
    val dimensions = mutableListOf<Float>()
    var currentArray: Any? = array

    while (currentArray != null) {
        when (currentArray) {
            is Array<*> -> {
                dimensions.add(currentArray.size.toFloat())
                currentArray = if (currentArray.isNotEmpty()) currentArray[0] else null
            }
            is FloatArray -> {
                dimensions.add(currentArray.size.toFloat())
                currentArray = null
            }
            else -> currentArray = null
        }
    }

    return dimensions
}

@Composable
fun CameraPreviewScreen() {
    val context = LocalContext.current
    val tfliteModel = remember { TFLiteModel(context) }
    val lensFacing = CameraSelector.LENS_FACING_BACK
    val lifecycleOwner = LocalLifecycleOwner.current
    val preview = Preview.Builder().build()
    val cameraxSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
    val imageCapture = remember {
        ImageCapture.Builder().build()
    }
    var handLandmarkerResult by remember { mutableStateOf<HandLandmarkerResult?>(null) }
    val handLandmarks3DArray = Array(2) { Array(21) { FloatArray(3) { 0f } } }

    val previewView = remember { PreviewView(context) }
    var imageWidth by remember { mutableStateOf(1) }
    var imageHeight by remember { mutableStateOf(1) }
    var predicted_out by remember { mutableStateOf("")}
    var label by remember { mutableStateOf("")}
    var myconfidence by remember { mutableStateOf<Float?>(null)}

    val handLandmarkerHelper = remember {
        HandLandmarkerHelper(
            runningMode = RunningMode.LIVE_STREAM,
            context = context,
            handLandmarkerHelperListener = object : HandLandmarkerHelper.LandmarkerListener {
                override fun onError(error: String, errorCode: Int) {
                    // Handle errors here
                    Log.e("HandLandmarker", "Error: $error")
                }

                override fun onResults(resultBundle: HandLandmarkerHelper.ResultBundle) {
                    val mpelapsedtime = resultBundle.inferenceTime
                    val handLandmarkerResults = resultBundle.results.firstOrNull()
                    handLandmarkerResults?.let { result ->
//                        if (result.landmarks().isNotEmpty()) {
                            handLandmarkerResult = handLandmarkerResults
//                        }

                        if (result.landmarks().size == 0){
                            for (j in 0 until 2){
                                for (i in 0 until 21){
                                    handLandmarks3DArray[j][i][0] = 0F
                                    handLandmarks3DArray[j][i][1] = 0F
                                    handLandmarks3DArray[j][i][2] = 0F
                                }
                            }

                        } else if (result.landmarks().size == 1){
                            for (handIndex in 0 until result.landmarks().size) {
                                val handedness = result.handednesses()[handIndex][0].categoryName()
                                val arrayIndex = if (handedness == "Right") 0 else 1
                                for (i in 0 until 21) {
                                    handLandmarks3DArray[if (arrayIndex == 0) 1 else 0][i][0] = 0F
                                    handLandmarks3DArray[if (arrayIndex == 0) 1 else 0][i][1] = 0F
                                    handLandmarks3DArray[if (arrayIndex == 0) 1 else 0][i][2] = 0F
                                }
                            }
                        }

                        for (handIndex in 0 until result.landmarks().size) {
                            val landmarks = result.landmarks()[handIndex]
                            val handedness = result.handednesses()[handIndex][0].categoryName()
                            val arrayIndex = if (handedness == "Right") 1 else 0

                            for (i in 0 until 21) {
                                handLandmarks3DArray[arrayIndex][i][0] = landmarks[i].x()
                                handLandmarks3DArray[arrayIndex][i][1] = landmarks[i].y()
                                handLandmarks3DArray[arrayIndex][i][2] = 0.0F
                            }
                        }
                        val wrappedoutput=Array(1){
                            handLandmarks3DArray
                        }
                        Log.d("coords", Arrays.deepToString(wrappedoutput))
                        val startTime = System.nanoTime()
                        val output = tfliteModel.runInference(wrappedoutput)
                        val endTime = System.nanoTime()
                        val elapsedTimeMs = (endTime - startTime) / 1_000_000 // Convert to milliseconds
                        val highestConfidence = output?.flatMap { it.toList() }?.maxOrNull()
                        val highestConfidenceIndex = output?.flatMap { it.toList() }?.indexOf(highestConfidence)
                        if (highestConfidenceIndex != null && highestConfidenceIndex >= 0) {
                            val labels = context.assets.open("labels.txt").bufferedReader().useLines { it.toList() }

                            if (highestConfidenceIndex < labels.size) {
                                val correspondingLabel = labels[highestConfidenceIndex]
                                predicted_out = "Confidence: $highestConfidence, Label: $correspondingLabel, Model Elapsed time: $elapsedTimeMs ms, MP Elapsed time : $mpelapsedtime"
                                Log.d("modeloutput", predicted_out)
                                label = correspondingLabel
                                myconfidence = highestConfidence
                            } else {
                                Log.e("modeloutput", "Index out of bounds for labels file")
                            }
                        } else {
                            Log.e("modeloutput", "Failed to find highest confidence or index")
                        }

                        Log.e("inferensi", "total inferensi = $elapsedTimeMs + $mpelapsedtime ms")

                    }
                }
            }
        )
    }

    LaunchedEffect(lensFacing) {
        val cameraProvider = context.getCameraProvider()
        cameraProvider.unbindAll()
        cameraProvider.bindToLifecycle(lifecycleOwner, cameraxSelector, preview, imageCapture)

        preview.setSurfaceProvider(previewView.surfaceProvider)

        val resolutionSelector = ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
            // Or use RATIO_16_9_FALLBACK_AUTO_STRATEGY for 16:9
            .setResolutionStrategy(ResolutionStrategy.HIGHEST_AVAILABLE_STRATEGY)
            .build()

        val imageAnalyzer = ImageAnalysis.Builder()
//            .setTargetResolution(Size(320, 320))
//            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setResolutionSelector(resolutionSelector)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            .also {
                it.setAnalyzer(
                    ContextCompat.getMainExecutor(context)
                ) { imageProxy ->
                    try {
                        handLandmarkerHelper.detectLiveStream(
                            imageProxy,
                            isFrontCamera = false
                        )
                        imageWidth = imageProxy.width
                        imageHeight = imageProxy.height

                    } catch (e: Exception) {
                        Log.e("ImageAnalysis", "Error converting image", e)
                    }
                    finally {
                        imageProxy.close()
                    }
                }
            }

        cameraProvider.bindToLifecycle(
            lifecycleOwner,
            cameraxSelector,
            preview,
            imageCapture,
            imageAnalyzer
        )

    }

    DisposableEffect(Unit) {
        onDispose {
            handLandmarkerHelper.clearHandLandmarker()
        }
    }

    HandLandmarkerView(
        previewView = previewView,
        handLandmarkerResult= handLandmarkerResult,
        imageWidth = imageWidth,
        imageHeight = imageHeight,
        label = predicted_out,
        predicted = label,
        confidence = myconfidence

    )



}


suspend fun Context.getCameraProvider(): ProcessCameraProvider =
    suspendCoroutine { continuation ->
        ProcessCameraProvider.getInstance(this).also { cameraProvider ->
            cameraProvider.addListener({
                continuation.resume(cameraProvider.get())
            }, ContextCompat.getMainExecutor(this))
        }
    }

@Composable
fun HandLandmarkerView(
    previewView: PreviewView,
    handLandmarkerResult: HandLandmarkerResult?,
    imageWidth: Int,
    imageHeight: Int,
    label: String = "un millon",
    predicted: String = "gada",
    confidence: Float? = 0.0f
) {



    Box(modifier = Modifier.fillMaxSize()) {
        // Camera preview
        AndroidView(
            factory = { previewView },
            modifier = Modifier.fillMaxSize()
        )

        Text(
            text = label,
            color = Color.White,
            modifier = Modifier
                .background(Color.Black.copy(alpha = 0.5f))
                .padding(top = 30.dp)
                .align(Alignment.TopCenter)
        )


        // Landmarks overlay
        if (handLandmarkerResult != null) {
//            val a = 1
//            if (handLandmarkerResult.handednesses().size == 0)
//            {
//                Canvas(modifier = Modifier
//                    .fillMaxSize()
//                    .background(Color.Transparent)){
//                    drawRect(color = Color.Transparent)
//                }
//            }
//            else
            Canvas(modifier = Modifier.fillMaxSize()) {

                val scaleFactor = minOf(
                    size.width / imageWidth.toFloat(),
                    size.height / imageHeight.toFloat()
                )

                val scaleX = size.width / imageWidth.toFloat()
                val scaleY = size.height / imageHeight.toFloat()




                // Draw landmarks
                handLandmarkerResult.landmarks().forEach { landmarks ->
                    // Draw points
                    landmarks.forEach { landmark ->
                        val x = landmark.x() * imageWidth * scaleX
                        val y = landmark.y() * imageHeight * scaleY

                        drawCircle(
                            color = Color.Yellow,
                            radius = 8f,
                            center = Offset(x, y)
                        )
                    }
//
//                    // Draw connections
                    HandLandmarker.HAND_CONNECTIONS.forEach { connection ->
                        val startLandmark = landmarks[connection.start()]
                        val endLandmark = landmarks[connection.end()]

                        drawLine(
                            color = Color(0xFFBB86FC), // Purple 200
                            start = Offset(
                                startLandmark.x() * imageWidth * scaleX,
                                startLandmark.y() * imageHeight * scaleY
                            ),
                            end = Offset(
                                endLandmark.x() * imageWidth * scaleX,
                                endLandmark.y() * imageHeight * scaleY
                            ),
                            strokeWidth = 8f
                        )
                    }
                }
            }



        }

    }

    if (confidence != null) {
        MainScreenUI(label = if (confidence > 0.85) predicted else "" )
    }
}






