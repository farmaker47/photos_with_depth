package com.soloupis.sample.photos_with_depth.fragments.segmentation

import android.app.Application
import android.content.Context
import android.graphics.*
import android.os.SystemClock
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import org.koin.core.KoinComponent
import org.koin.core.get
import java.io.IOException


class DepthAndStyleViewModel(application: Application) :
    AndroidViewModel(application),
    KoinComponent {

    private lateinit var outputBitmapBlack: Bitmap
    private lateinit var outputBitmapGray: Bitmap
    var startTime: Long = 0L
    var inferenceTime = 0L
    lateinit var scaledBitmapObject: Bitmap

    var stylename = String()
    var seekBarProgress: Float = 0F

    private var _currentList: ArrayList<String> = ArrayList()
    val currentList: ArrayList<String>
        get() = _currentList

    private val _totalTimeInference = MutableLiveData<Int>()
    val totalTimeInference: LiveData<Int>
        get() = _totalTimeInference

    private val _styledBitmap = MutableLiveData<Bitmap>()
    val styledBitmap: LiveData<Bitmap>
        get() = _styledBitmap

    private val _inferenceDone = MutableLiveData<Boolean>()
    val inferenceDone: LiveData<Boolean>
        get() = _inferenceDone

    val depthAndStyleModelExecutor: DepthAndStyleModelExecutor

    init {

        stylename = "mona.JPG"

        _currentList.addAll(application.assets.list("thumbnails")!!)

        depthAndStyleModelExecutor = get()

    }

    fun setStyleName(string: String) {
        stylename = string
    }

    fun performDepthAndStyleProcedure(
        bitmap: Bitmap,
        context: Context
    ): Triple<Bitmap, Bitmap, Long> {
        try {
            // Initialization
            startTime = SystemClock.uptimeMillis()

            // Run inference
            val (output1, output2) = depthAndStyleModelExecutor.executeProcedureForPhotosWithDepth(
                bitmap,
                context
            )

            outputBitmapGray = output1
            outputBitmapBlack = output2

            inferenceTime = SystemClock.uptimeMillis() - startTime
        } catch (e: IOException) {
            Log.e("Depth", "Error: ", e)
        }

        return Triple(outputBitmapGray, outputBitmapBlack, inferenceTime)
    }

    fun cropBitmapWithMask(original: Bitmap, mask: Bitmap?, string: String): Bitmap? {
        if (original == null || mask == null
        ) {
            return null
        }
        Log.i("ORIGINAL_WIDTH", original.width.toString())
        Log.i("ORIGINAL_HEIGHT", original.height.toString())
        Log.i("MASK_WIDTH", original.width.toString())
        Log.i("MASK_HEIGHT", original.height.toString())
        val w = original.width
        val h = original.height
        if (w <= 0 || h <= 0) {
            return null
        }

        // Generate colored foreground with transparent background
        val cropped: Bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(cropped)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG)
        paint.xfermode = PorterDuffXfermode(PorterDuff.Mode.DST_IN)
        canvas.drawBitmap(original, 0f, 0f, null)
        canvas.drawBitmap(mask, 0f, 0f, paint)
        paint.xfermode = null

        // Generate final bitmap with colored foreground and B/W background
        val croppedFinal: Bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvasFinal = Canvas(croppedFinal)
        val paintFinal = Paint(Paint.ANTI_ALIAS_FLAG)
        paintFinal.xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC_OVER)
        canvasFinal.drawBitmap(androidGrayScale(original), 0f, 0f, null)
        canvasFinal.drawBitmap(cropped, 0f, 0f, paint)
        paintFinal.xfermode = null

        _styledBitmap.postValue(croppedFinal)
        _inferenceDone.postValue(true)

        return croppedFinal
    }

    private fun androidGrayScale(bmpOriginal: Bitmap): Bitmap {
        val height: Int = bmpOriginal.height
        val width: Int = bmpOriginal.width
        val bmpGrayscale = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmpGrayscale)
        val paint = Paint()

        // The conversion is based on OpenCV RGB to GRAY conversion
        // https://docs.opencv.org/master/de/d25/imgproc_color_conversions.html#color_convert_rgb_gray
        // The luminance of each pixel is calculated as the weighted sum of the 3 RGB values
        // Y = 0.299R + 0.587G + 0.114B
        val matrix = floatArrayOf(
            0.299f, 0.587f, 0.114f, 0.0f, 0.0f,
            0.299f, 0.587f, 0.114f, 0.0f, 0.0f,
            0.299f, 0.587f, 0.114f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f, 0.0f
        )
        val colorMatrixFilter = ColorMatrixColorFilter(matrix)

        paint.colorFilter = colorMatrixFilter
        canvas.drawBitmap(bmpOriginal, 0f, 0f, paint)
        return bmpGrayscale
    }

    fun cropBitmapWithMaskForStyle(original: Bitmap, mask: Bitmap?): Bitmap? {
        if (original == null || mask == null
        ) {
            return null
        }
        val w = original.width
        val h = original.height
        if (w <= 0 || h <= 0) {
            return null
        }

        val scaledBitmap = Bitmap.createScaledBitmap(
            mask,
            w,
            h, true
        )

        val cropped = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(cropped)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG)
        paint.xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC_OVER)
        canvas.drawBitmap(original, 0f, 0f, null)
        canvas.drawBitmap(scaledBitmap, 0f, 0f, paint)
        paint.xfermode = null
        return cropped
    }

    override fun onCleared() {
        super.onCleared()
        depthAndStyleModelExecutor.close()
    }

}