package com.soloupis.sample.photos_with_depth.fragments.segmentation

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.soloupis.sample.photos_with_depth.utils.ImageUtils
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.image.TensorImage
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

data class ModelExecutionResult(
    val styledImage: Bitmap,
    val preProcessTime: Long = 0L,
    val stylePredictTime: Long = 0L,
    val styleTransferTime: Long = 0L,
    val postProcessTime: Long = 0L,
    val totalExecutionTime: Long = 0L,
    val executionLog: String = "",
    val errorMessage: String = ""
)

@SuppressWarnings("GoodTime")
class DepthAndStyleModelExecutor(
    context: Context,
    private var useGPU: Boolean = false
) {

    private var numberThreads = 4
    private var fullExecutionTime = 0L
    private var preProcessTime = 0L
    private var stylePredictTime = 0L
    private var styleTransferTime = 0L
    private var postProcessTime = 0L
    private var interprerDepth: Interpreter
    private lateinit var gpuDelegate: GpuDelegate

    companion object {
        private const val TAG = "PhotosWithDepthProcedure"
        private const val CONTENT_IMAGE_SIZE = 384
        private const val DEPTH_MODEL = "3D_depth.tflite"
    }

    init {

        interprerDepth = getInterpreter(context, DEPTH_MODEL, useGPU)

    }

    // Function for ML Binding
    fun executeProcedureForPhotosWithDepth(
        contentImage: Bitmap,
        context: Context
    ): IntArray {
        try {
            Log.i(TAG, "running models")

            fullExecutionTime = SystemClock.uptimeMillis()

            preProcessTime = SystemClock.uptimeMillis()
            // Creates inputs for reference.

            val loadedImage = TensorImage.fromBitmap(contentImage).tensorBuffer
            preProcessTime = SystemClock.uptimeMillis() - preProcessTime

            stylePredictTime = SystemClock.uptimeMillis()
            // Runs model inference and gets result.
            Log.d(TAG, "Style Predict Time to run: $stylePredictTime")
            //val outputsPredict = interprerDepth.run...........
            stylePredictTime = SystemClock.uptimeMillis() - stylePredictTime
            Log.d(TAG, "Style Predict Time to run: $stylePredictTime")

            styleTransferTime = SystemClock.uptimeMillis()
            // Runs model inference and gets result.
            styleTransferTime = SystemClock.uptimeMillis() - styleTransferTime
            Log.d(TAG, "Style apply Time to run: $styleTransferTime")

            postProcessTime = SystemClock.uptimeMillis()
            postProcessTime = SystemClock.uptimeMillis() - postProcessTime

            fullExecutionTime = SystemClock.uptimeMillis() - fullExecutionTime
            Log.d(TAG, "Time to run everything: $fullExecutionTime")

            return intArrayOf()//outputsPredict.arrayOutputAsTensorBuffer.intArray
        } catch (e: Exception) {
            val exceptionLog = "something went wrong: ${e.message}"
            Log.e("EXECUTOR", exceptionLog)

            /*val emptyBitmap =
                ImageUtils.createEmptyBitmap(
                    CONTENT_IMAGE_SIZE,
                    CONTENT_IMAGE_SIZE
                )*/
            return intArrayOf()
        }
    }

    @Throws(IOException::class)
    private fun loadModelFile(context: Context, modelFile: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelFile)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        val retFile = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        fileDescriptor.close()
        return retFile
    }

    @Throws(IOException::class)
    private fun getInterpreter(
        context: Context,
        modelName: String,
        useGpu: Boolean
    ): Interpreter {
        val tfliteOptions = Interpreter.Options()
        if (useGpu) {
            gpuDelegate = GpuDelegate()
            tfliteOptions.addDelegate(gpuDelegate)

            // Create the Delegate instance.
            /*try {
                gpuDelegate = HexagonDelegate(context)
                tfliteOptions.addDelegate(gpuDelegate)
            } catch (e: Exception) {
                // Hexagon delegate is not supported on this device.
                Log.e("HEXAGON", e.toString())
            }*/

            //val delegate =
            //GpuDelegate(GpuDelegate.Options().setQuantizedModelsAllowed(true))
        }

        tfliteOptions.setNumThreads(numberThreads)
        //tfliteOptions.setUseXNNPACK(true)
        return Interpreter(loadModelFile(context, modelName), tfliteOptions)
    }

    private fun formatExecutionLog(): String {
        val sb = StringBuilder()
        sb.append("Input Image Size: $CONTENT_IMAGE_SIZE x $CONTENT_IMAGE_SIZE\n")
        sb.append("GPU enabled: $useGPU\n")
        sb.append("Number of threads: $numberThreads\n")
        sb.append("Pre-process execution time: $preProcessTime ms\n")
        sb.append("Predicting style execution time: $stylePredictTime ms\n")
        sb.append("Transferring style execution time: $styleTransferTime ms\n")
        sb.append("Post-process execution time: $postProcessTime ms\n")
        sb.append("Full execution time: $fullExecutionTime ms\n")
        return sb.toString()
    }

    fun close() {
        interprerDepth.close()
    }
}
