package com.example.facedetector
import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.nio.channels.FileChannel

class TFLiteModel(context: Context) {
    private val interpreter: Interpreter
    private val TAG = "TFLiteModel"

    init {
        val assetFileDescriptor = context.assets.openFd("datasetsemogafinalv2.tflite")
        val fileInputStream = assetFileDescriptor.createInputStream()
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        val model = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        interpreter = Interpreter(model)

        val inputShape = interpreter.getInputTensor(0).shape()
        val outputShape = interpreter.getOutputTensor(0).shape()
        Log.d(TAG, "Model input shape: ${inputShape.contentToString()}")
        Log.d(TAG, "Model output shape: ${outputShape.contentToString()}")
    }


    fun runInference(inputArray: Array<Array<Array<FloatArray>>>): Array<FloatArray>? {
//        val dim = getDimensions(inputArray)
//        Log.d(TAG, "input input shape: $dim")
        try {
            val output = Array(1){
                FloatArray(21)
            }
            interpreter.run(inputArray, output)

            Log.d(TAG, "Inference successful: ${output.contentToString()}")
            return output
        } catch (e: Exception) {
            Log.e(TAG, "Error during inference: ${e.message}")
            e.printStackTrace()
            return null
        }
    }
}