package com.example.rom

import PoseEstimationHelper
import android.content.ContentValues
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Build.VERSION.SDK_INT
import android.os.Build.VERSION_CODES.TIRAMISU
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.view.PixelCopy
import android.view.View
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import coil.load
import com.example.rom.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import timber.log.Timber
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.OutputStream

// TODO 오차 보정(문서 및 레포 참고)
// TODO 사진 촬영시 Pose Estimation 이 사물을 제대로 인식하지 못하는 케이스 해결(신뢰할 수 있는 값만 선별)
class MainActivity : AppCompatActivity() {
    companion object {
        private const val MODEL_PATH = "movenet_lightning.tflite"
        const val EXTRA_POSE_ESTIMATION_RESULT = "pose_estimation_result"
        const val EXTRA_VALIDATION_MESSAGE = "validation_message"
    }

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraActivityResultLauncher: ActivityResultLauncher<Intent>

    private val viewModel by viewModels<MainViewModel>()

    private val pickMedia = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) {
            processImageForPoseEstimation(uri)
        } else {
            Timber.d("No media selected")
        }
    }

    private val tflite by lazy {
        Interpreter(
            FileUtil.loadMappedFile(this, MODEL_PATH),
            Interpreter.Options().addDelegate(nnApiDelegate),
        )
    }

    private val nnApiDelegate by lazy {
        NnApiDelegate()
    }

    private val tfImageProcessor by lazy {
        ImageProcessor.Builder()
            .add(ResizeOp(PoseEstimationHelper.IMAGE_SIZE, PoseEstimationHelper.IMAGE_SIZE, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 1f))
            .build()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraActivityResultLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            val resultData: ResultData?
            if (result.resultCode == RESULT_OK) {
                resultData = if (SDK_INT >= TIRAMISU) {
                    result.data?.getParcelableExtra(EXTRA_POSE_ESTIMATION_RESULT, ResultData::class.java)
                } else {
                    result.data?.getParcelableExtra(EXTRA_POSE_ESTIMATION_RESULT)
                }
                viewModel.setResultData(resultData!!)

                val validationMessage: String? = result.data?.getStringExtra(EXTRA_VALIDATION_MESSAGE)
                viewModel.setValidationMessage(validationMessage!!)
            }
        }

        binding.btnCaptureCamera.setOnClickListener {
            cameraActivityResultLauncher.launch(Intent(this, CameraActivity::class.java))
        }

//        binding.btnGallery.setOnClickListener {
//            pickMedia.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
//        }

        binding.btnSave.setOnClickListener {
            captureAndSaveScreen()
        }

        lifecycleScope.launch {
            lifecycle.repeatOnLifecycle(Lifecycle.State.STARTED) {
                launch {
                    viewModel.resultData.collect {
                        if (it.leftAngleBefore == 0f && it.rightAngleBefore == 0f && it.imageByteArray == null) {
                            binding.apply {
                                ivResultImage.visibility = View.GONE

                                tvResult.visibility = View.GONE

                                tvLeftAngleBefore.visibility = View.GONE
                                tvLeftAngleValueBefore.visibility = View.GONE

                                tvRightAngleBefore.visibility = View.GONE
                                tvRightAngleValueBefore.visibility = View.GONE

                                tvLeftShoulderAngle.visibility = View.GONE
                                tvLeftShoulderAngleValue.visibility = View.GONE

                                tvRightShoulderAngle.visibility = View.GONE
                                tvRightShoulderAngleValue.visibility = View.GONE

                                tvLeftElbowAngle.visibility = View.GONE
                                tvLeftElbowAngleValue.visibility = View.GONE

                                tvRightElbowAngle.visibility = View.GONE
                                tvRightElbowAngleValue.visibility = View.GONE

                                tvLeftAngleAfter.visibility = View.GONE
                                tvLeftAngleValueAfter.visibility = View.GONE

                                tvRightAngleAfter.visibility = View.GONE
                                tvRightAngleValueAfter.visibility = View.GONE

                                btnSave.visibility = View.GONE
                            }
                        } else {
                            binding.apply {
                                ivResultImage.load(it.imageByteArray)
                                ivResultImage.visibility = View.VISIBLE

                                tvResult.visibility = View.VISIBLE

                                tvLeftAngleBefore.visibility = View.VISIBLE
                                tvLeftAngleValueBefore.text =
                                    String.format("%.1f°(%.1f° ~ %.1f°)", it.leftAngleBefore, it.leftAngleBefore - 2.5, it.leftAngleBefore + 2.5)
                                tvLeftAngleValueBefore.visibility = View.VISIBLE

                                tvRightAngleBefore.visibility = View.VISIBLE
                                tvRightAngleValueBefore.text =
                                    String.format("%.1f°(%.1f° ~ %.1f°)", it.rightAngleBefore, it.rightAngleBefore - 2.5, it.rightAngleBefore + 2.5)
                                tvRightAngleValueBefore.visibility = View.VISIBLE

                                tvLeftShoulderAngle.visibility = View.VISIBLE
                                tvLeftShoulderAngleValue.text = String.format("%.1f°", it.leftShoulderAngle)
                                tvLeftShoulderAngleValue.visibility = View.VISIBLE

                                tvRightShoulderAngle.visibility = View.VISIBLE
                                tvRightShoulderAngleValue.text = String.format("%.1f°", it.rightShoulderAngle)
                                tvRightShoulderAngleValue.visibility = View.VISIBLE

                                tvLeftElbowAngle.visibility = View.VISIBLE
                                tvLeftElbowAngleValue.text = String.format("%.1f°", it.leftElbowAngle)
                                tvLeftElbowAngleValue.visibility = View.VISIBLE

                                tvRightElbowAngle.visibility = View.VISIBLE
                                tvRightElbowAngleValue.text = String.format("%.1f°", it.rightElbowAngle)
                                tvRightElbowAngleValue.visibility = View.VISIBLE

                                ivResultImage.load(it.imageByteArray)
                                ivResultImage.visibility = View.VISIBLE

                                tvResult.visibility = View.VISIBLE

                                tvLeftAngleAfter.visibility = View.VISIBLE
                                tvLeftAngleValueAfter.text =
                                    String.format("%.1f°(%.1f° ~ %.1f°)", it.leftAngleAfter, it.leftAngleAfter - 2.5, it.leftAngleAfter + 2.5)
                                tvLeftAngleValueAfter.visibility = View.VISIBLE

                                tvRightAngleAfter.visibility = View.VISIBLE
                                tvRightAngleValueAfter.text =
                                    String.format("%.1f°(%.1f° ~ %.1f°)", it.rightAngleAfter, it.rightAngleAfter - 2.5, it.rightAngleAfter + 2.5)
                                tvRightAngleValueAfter.visibility = View.VISIBLE

                                btnSave.visibility = View.VISIBLE
                            }
                        }
                    }
                }

                launch {
                    viewModel.validationMessage.collect {
                        if (it.isNotEmpty()) {
                            showErrorDialog(it)

                            binding.btnCaptureCamera.text = "다시 촬영"
                        }
                    }
                }
            }
        }
    }

    private fun captureAndSaveScreen() {
        val rootView = window.decorView.rootView
        rootView.isDrawingCacheEnabled = true
        val bitmap = Bitmap.createBitmap(rootView.drawingCache)
        rootView.isDrawingCacheEnabled = false

        val fileName = "ROM_${System.currentTimeMillis()}.jpg"
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            put(MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/ROM")
        }

        var outputStream: OutputStream? = null
        var uri: Uri?
        try {
            uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
            outputStream = uri?.let { contentResolver.openOutputStream(it) }
            outputStream?.use {
                bitmap.compress(Bitmap.CompressFormat.JPEG, 100, it)
            }
            Toast.makeText(this, "사진이 저장되었습니다.", Toast.LENGTH_SHORT).show()
        } catch (e: IOException) {
            e.printStackTrace()
            Toast.makeText(this, "사진 저장을 실패하였습니다.", Toast.LENGTH_SHORT).show()
        } finally {
            outputStream?.close()
        }
    }

    private fun processImageForPoseEstimation(uri: Uri) {
        lifecycleScope.launch {
            val bitmap = withContext(Dispatchers.IO) {
                contentResolver.openInputStream(uri)?.use {
                    BitmapFactory.decodeStream(it)
                }
            }

            bitmap?.let {
                val poseEstimationHelper = PoseEstimationHelper(tflite) // tflite 객체는 적절히 초기화 필요
                val tfImage = TensorImage.fromBitmap(it)
                val processedImage = tfImageProcessor.process(tfImage) // tfImageProcessor는 적절히 초기화 필요

                val predictions = poseEstimationHelper.predict(processedImage)
                val poseEstimationBitmap = drawPoseEstimation(it, predictions)

                val (armsAngle, validationMessage) = viewModel.calculateArmAngle(predictions)
                if (armsAngle != null) {
                    val imageByteArray = bitmapToByteArray(poseEstimationBitmap)
                    val resultData = ResultData(
                        armsAngle.leftAngleBefore,
                        armsAngle.rightAngleBefore,
                        armsAngle.leftShoulderAngle,
                        armsAngle.rightShoulderAngle,
                        armsAngle.leftElbowAngle,
                        armsAngle.rightElbowAngle,
                        armsAngle.leftAngleAfter,
                        armsAngle.rightAngleAfter,
                        imageByteArray,
                    )
                    viewModel.setResultData(resultData)
                    viewModel.setValidationMessage(validationMessage)
                }
            }
        }
    }

    private fun bitmapToByteArray(bitmap: Bitmap): ByteArray {
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
        return stream.toByteArray()
    }


    private fun drawPoseEstimation(bitmap: Bitmap, predictions: List<PoseEstimationHelper.PosePrediction>): Bitmap {
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)

        val paint = Paint()
        paint.strokeWidth = 2f
        paint.style = Paint.Style.STROKE

        predictions.forEach { prediction ->
            val keyPoints = prediction.keypoints
            val connections = listOf(
                Triple(PoseEstimationHelper.BodyPart.LEFT_SHOULDER, PoseEstimationHelper.BodyPart.RIGHT_SHOULDER, Color.MAGENTA),
                Triple(PoseEstimationHelper.BodyPart.LEFT_SHOULDER, PoseEstimationHelper.BodyPart.LEFT_ELBOW, Color.MAGENTA),
                Triple(PoseEstimationHelper.BodyPart.LEFT_ELBOW, PoseEstimationHelper.BodyPart.LEFT_WRIST, Color.MAGENTA),
                Triple(PoseEstimationHelper.BodyPart.RIGHT_SHOULDER, PoseEstimationHelper.BodyPart.RIGHT_ELBOW, Color.MAGENTA),
                Triple(PoseEstimationHelper.BodyPart.RIGHT_ELBOW, PoseEstimationHelper.BodyPart.RIGHT_WRIST, Color.MAGENTA),
                Triple(PoseEstimationHelper.BodyPart.LEFT_SHOULDER, PoseEstimationHelper.BodyPart.LEFT_HIP, Color.MAGENTA),
                Triple(PoseEstimationHelper.BodyPart.RIGHT_SHOULDER, PoseEstimationHelper.BodyPart.RIGHT_HIP, Color.MAGENTA),
                Triple(PoseEstimationHelper.BodyPart.LEFT_HIP, PoseEstimationHelper.BodyPart.RIGHT_HIP, Color.MAGENTA),
                Triple(PoseEstimationHelper.BodyPart.LEFT_HIP, PoseEstimationHelper.BodyPart.LEFT_KNEE, Color.MAGENTA),
                Triple(PoseEstimationHelper.BodyPart.LEFT_KNEE, PoseEstimationHelper.BodyPart.LEFT_ANKLE, Color.MAGENTA),
                Triple(PoseEstimationHelper.BodyPart.RIGHT_HIP, PoseEstimationHelper.BodyPart.RIGHT_KNEE, Color.MAGENTA),
                Triple(PoseEstimationHelper.BodyPart.RIGHT_KNEE, PoseEstimationHelper.BodyPart.RIGHT_ANKLE, Color.MAGENTA),
            )

            connections.forEach { (start, end, color) ->
                val startPoint = keyPoints.find { it.bodyPart == start }?.position
                val endPoint = keyPoints.find { it.bodyPart == end }?.position
                if (startPoint != null && endPoint != null) {
                    paint.color = color
                    canvas.drawLine(
                        startPoint.x * bitmap.width, startPoint.y * bitmap.height,
                        endPoint.x * bitmap.width, endPoint.y * bitmap.height, paint,
                    )
                }
            }

            // 키포인트 그리기
            paint.style = Paint.Style.FILL
            keyPoints.forEach { keypoint ->
//                paint.color = when (keypoint.bodyPart) {
//                    PoseEstimationHelper.BodyPart.LEFT_SHOULDER,
//                    PoseEstimationHelper.BodyPart.LEFT_ELBOW,
//                    PoseEstimationHelper.BodyPart.LEFT_HIP,
//                    -> Color.MAGENTA
//
//                    PoseEstimationHelper.BodyPart.RIGHT_SHOULDER,
//                    PoseEstimationHelper.BodyPart.RIGHT_ELBOW,
//                    PoseEstimationHelper.BodyPart.RIGHT_HIP,
//                    -> Color.YELLOW
//
//                    else -> Color.CYAN
//                }
                paint.color = Color.MAGENTA
                canvas.drawCircle(
                    keypoint.position.x * bitmap.width,
                    keypoint.position.y * bitmap.height,
                    2f, paint,
                )
            }
        }

        return mutableBitmap
    }

    private fun showErrorDialog(message: String) {
        AlertDialog.Builder(this)
            .setTitle("경고")
            .setMessage(message)
            .setPositiveButton("확인") { dialog, _ ->
                dialog.dismiss()
            }
            .show()
    }
}
