package com.example.rom

import PoseEstimationHelper
import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.os.Bundle
import android.util.Log
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.lifecycleScope
import com.example.rom.databinding.ActivityCameraBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import timber.log.Timber
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.random.Random

/** 카메라를 표시하고 들어오는 프레임에 대해 객체 감지를 수행하는 액티비티 */
class CameraActivity : AppCompatActivity() {
    companion object {
        private const val MODEL_PATH = "movenet_lightning.tflite"
    }

    private lateinit var binding: ActivityCameraBinding

    // 카메라 이미지를 저장할 버퍼
    private lateinit var bitmapBuffer: Bitmap

    // 백그라운드 스레드에서 이미지 처리를 실행할 실행기
    private val executor = Executors.newSingleThreadExecutor()

    // 앱에서 필요한 권한 목록
    private val permissions = listOf(Manifest.permission.CAMERA)

    // 권한 요청을 위한 랜덤 코드
    private val permissionsRequestCode = Random.nextInt(0, 10000)

    private var cameraProvider: ProcessCameraProvider? = null

    // 카메라 후면 방향
    private var lensFacing: Int = CameraSelector.LENS_FACING_BACK

    // 카메라 전면 방향 여부
    private val isFrontFacing get() = lensFacing == CameraSelector.LENS_FACING_FRONT

    // 이미지 분석 중지 여부
    private var pauseAnalysis = false

    // 이미지 회전 각도
    private var imageRotationDegrees: Int = 0

    private var isPoseEstimationEnabled = false

    // TensorFlow Lite 이미지 처리기 초기화
    private val tfImageProcessor by lazy {
        ImageProcessor.Builder()
            .add(ResizeOp(PoseEstimationHelper.IMAGE_SIZE, PoseEstimationHelper.IMAGE_SIZE, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 1f))
            .build()
    }

    // NNAPI 델리게이트 초기화
    private val nnApiDelegate by lazy {
        NnApiDelegate()
    }

    // TensorFlow Lite 인터프리터 초기화
    private val tflite by lazy {
        Interpreter(
            FileUtil.loadMappedFile(this, MODEL_PATH),
            Interpreter.Options().addDelegate(nnApiDelegate),
        )
    }

    private val poseEstimationHelper by lazy {
        val helper = PoseEstimationHelper(tflite)
        val inputTensor = tflite.getInputTensor(0)
        val outputTensor = tflite.getOutputTensor(0)
        Timber.d("Input tensor shape: ${inputTensor.shape().contentToString()}")
        Timber.d("Input tensor type: ${inputTensor.dataType()}")
        Timber.d("Output tensor shape: ${outputTensor.shape().contentToString()}")
        Timber.d("Output tensor type: ${outputTensor.dataType()}")
        helper
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val inputShape = tflite.getInputTensor(0).shape()
        Timber.d("Model input shape: ${inputShape.contentToString()}")

        binding.btnCaptureCamera.setOnClickListener {
            it.isEnabled = false

            if (pauseAnalysis) {
                // 분석 재개
                pauseAnalysis = false
                binding.imagePredicted.visibility = View.GONE
            } else {
                // 현재 프레임 캡처 및 분석
                pauseAnalysis = true
                val matrix = Matrix().apply {
                    postRotate(imageRotationDegrees.toFloat())
                    if (isFrontFacing) postScale(-1f, 1f)
                }
                val uprightImage = Bitmap.createBitmap(
                    bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height, matrix, true,
                )

                val tfImage = TensorImage.fromBitmap(uprightImage)
                val processedImage = tfImageProcessor.process(tfImage)

                Timber.d("Processed image size: ${processedImage.width}x${processedImage.height}, " + "buffer size: ${processedImage.buffer.capacity()} bytes")

                val prediction = poseEstimationHelper.predict(processedImage)

                Timber.d("Predictions: $prediction")

                prediction.let { posePrediction ->
                    val poseEstimationBitmap = drawPoseEstimation(uprightImage, posePrediction)
                    binding.imagePredicted.setImageBitmap(poseEstimationBitmap)
                    binding.imagePredicted.visibility = View.VISIBLE

                    val armsAngle = calculateArmAngle(posePrediction)
                    if (armsAngle != null) {
                        val imageByteArray = bitmapToByteArray(poseEstimationBitmap)
                        val resultIntent = Intent().apply {
                            putExtra(MainActivity.EXTRA_POSE_ESTIMATION_RESULT, ResultData(armsAngle.leftAngle, armsAngle.rightAngle, imageByteArray))
                        }
                        setResult(RESULT_OK, resultIntent)
                        finish()
                    }
                }

                it.isEnabled = true
            }
        }

        // 카메라 전환 버튼 리스너 설정
        binding.btnSwitchCamera.setOnClickListener {
            lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
                CameraSelector.LENS_FACING_BACK
            } else {
                CameraSelector.LENS_FACING_FRONT
            }
            bindCameraUseCases()
        }

        binding.btnTogglePoseEstimation.setOnClickListener {
            togglePoseEstimation()
        }

        // 초기 상태 설정
        updatePoseEstimationButton()
    }

    private fun drawPoseEstimation(bitmap: Bitmap, predictions: List<PoseEstimationHelper.PosePrediction>): Bitmap {
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)

        // 전면 카메라일 경우 캔버스를 수평으로 뒤집음
        if (isFrontFacing) {
            canvas.scale(1f, -1f, bitmap.width / 2f, bitmap.height / 2f)
        }

        val paint = Paint()
        paint.strokeWidth = 2f
        paint.style = Paint.Style.STROKE

        predictions.forEach { prediction ->
            val keyPoints = prediction.keypoints
            val connections = listOf(
                Triple(PoseEstimationHelper.BodyPart.LEFT_SHOULDER, PoseEstimationHelper.BodyPart.RIGHT_SHOULDER, Color.CYAN),
                Triple(PoseEstimationHelper.BodyPart.LEFT_SHOULDER, PoseEstimationHelper.BodyPart.LEFT_ELBOW, Color.MAGENTA),
                Triple(PoseEstimationHelper.BodyPart.LEFT_ELBOW, PoseEstimationHelper.BodyPart.LEFT_WRIST, Color.CYAN),
                Triple(PoseEstimationHelper.BodyPart.RIGHT_SHOULDER, PoseEstimationHelper.BodyPart.RIGHT_ELBOW, Color.YELLOW),
                Triple(PoseEstimationHelper.BodyPart.RIGHT_ELBOW, PoseEstimationHelper.BodyPart.RIGHT_WRIST, Color.CYAN),
                Triple(PoseEstimationHelper.BodyPart.LEFT_SHOULDER, PoseEstimationHelper.BodyPart.LEFT_HIP, Color.MAGENTA),
                Triple(PoseEstimationHelper.BodyPart.RIGHT_SHOULDER, PoseEstimationHelper.BodyPart.RIGHT_HIP, Color.YELLOW),
                Triple(PoseEstimationHelper.BodyPart.LEFT_HIP, PoseEstimationHelper.BodyPart.RIGHT_HIP, Color.CYAN),
                Triple(PoseEstimationHelper.BodyPart.LEFT_HIP, PoseEstimationHelper.BodyPart.LEFT_KNEE, Color.CYAN),
                Triple(PoseEstimationHelper.BodyPart.LEFT_KNEE, PoseEstimationHelper.BodyPart.LEFT_ANKLE, Color.CYAN),
                Triple(PoseEstimationHelper.BodyPart.RIGHT_HIP, PoseEstimationHelper.BodyPart.RIGHT_KNEE, Color.CYAN),
                Triple(PoseEstimationHelper.BodyPart.RIGHT_KNEE, PoseEstimationHelper.BodyPart.RIGHT_ANKLE, Color.CYAN),
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
                paint.color = when (keypoint.bodyPart) {
                    PoseEstimationHelper.BodyPart.LEFT_SHOULDER,
                    PoseEstimationHelper.BodyPart.LEFT_ELBOW,
                    PoseEstimationHelper.BodyPart.LEFT_HIP,
                    -> Color.MAGENTA

                    PoseEstimationHelper.BodyPart.RIGHT_SHOULDER,
                    PoseEstimationHelper.BodyPart.RIGHT_ELBOW,
                    PoseEstimationHelper.BodyPart.RIGHT_HIP,
                    -> Color.YELLOW

                    else -> Color.CYAN
                }
                canvas.drawCircle(
                    keypoint.position.x * bitmap.width,
                    keypoint.position.y * bitmap.height,
                    2f, paint,
                )
            }
        }

        return mutableBitmap
    }

    private fun bitmapToByteArray(bitmap: Bitmap): ByteArray {
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
        return stream.toByteArray()
    }

    override fun onDestroy() {
        // 모든 남은 분석 작업 종료
        executor.apply {
            shutdown()
            awaitTermination(1000, TimeUnit.MILLISECONDS)
        }

        // TensorFlow Lite 리소스 해제
        tflite.close()
        nnApiDelegate.close()

        super.onDestroy()
    }

    @SuppressLint("UnsafeExperimentalUsageError")
    private fun bindCameraUseCases() = binding.viewFinder.post {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener(
            {
                cameraProvider = cameraProviderFuture.get()

                // 카메라 프리뷰 설정
                val preview = Preview.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                    .setTargetRotation(binding.viewFinder.display.rotation)
                    .build()

                // 실시간으로 프레임을 처리할 이미지 분석 사용 사례 설정
                val imageAnalysis = ImageAnalysis.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                    .setTargetRotation(binding.viewFinder.display.rotation)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()

                var frameCounter = 0
                var lastFpsTimestamp = System.currentTimeMillis()

                imageAnalysis.setAnalyzer(
                    executor,
                    ImageAnalysis.Analyzer { image ->
                        if (!::bitmapBuffer.isInitialized) {
                            imageRotationDegrees = image.imageInfo.rotationDegrees
                            bitmapBuffer = Bitmap.createBitmap(
                                image.width, image.height, Bitmap.Config.ARGB_8888,
                            )
                        }

                        if (pauseAnalysis) {
                            image.close()
                            return@Analyzer
                        }

                        image.use {
                            bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer)
                        }

                        val matrix = Matrix().apply {
                            postRotate(imageRotationDegrees.toFloat())
                            if (isFrontFacing) {
                                postScale(1f, -1f, image.width / 2f, image.height / 2f)
                            }
                        }

                        val rotatedBitmap = Bitmap.createBitmap(
                            bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height, matrix, true,
                        )

                        if (isPoseEstimationEnabled) {
                            val tfImage = TensorImage.fromBitmap(rotatedBitmap)
                            val processedImage = tfImageProcessor.process(tfImage)

                            val predictions = PoseEstimationHelper(tflite).predict(processedImage)

                            lifecycleScope.launch(Dispatchers.Main) {
                                if (predictions.isNotEmpty()) {
                                    val poseEstimationBitmap = drawPoseEstimation(rotatedBitmap, predictions)
                                    binding.imagePredicted.setImageBitmap(poseEstimationBitmap)
                                    binding.imagePredicted.visibility = View.VISIBLE

                                    val armsAngle = calculateArmAngle(predictions)
                                    if (armsAngle != null) {
                                        binding.textPrediction.text = "Left Angle: ${armsAngle.leftAngle}, Right Angle: ${armsAngle.rightAngle}"
                                        binding.textPrediction.visibility = View.VISIBLE
                                    }
                                }
                            }

                            // 전체 파이프라인의 FPS 계산
                            val frameCount = 10
                            if (++frameCounter % frameCount == 0) {
                                frameCounter = 0
                                val now = System.currentTimeMillis()
                                val delta = now - lastFpsTimestamp
                                val fps = 1000 * frameCount.toFloat() / delta
                                Timber.d("FPS: ${"%.02f".format(fps)} with tensorSize: ${tfImage.width} x ${tfImage.height}")
                                lastFpsTimestamp = now
                            }
                        } else {
                            lifecycleScope.launch(Dispatchers.Main) {
                                binding.imagePredicted.visibility = View.GONE
                                binding.textPrediction.visibility = View.GONE
                            }
                        }
                    },
                )

                // 렌즈 방향 설정
                val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()

                try {
                    // 기존 사용 사례 언바인딩
                    cameraProvider?.unbindAll()

                    // 새로운 사용 사례 바인딩
                    cameraProvider?.bindToLifecycle(
                        this as LifecycleOwner,
                        cameraSelector,
                        preview,
                        imageAnalysis,
                    )

                    // 프리뷰 사용 사례를 뷰와 연결
                    preview.setSurfaceProvider(binding.viewFinder.surfaceProvider)
                } catch (exc: Exception) {
                    Timber.e("Use case binding failed", exc)
                }
            },
            ContextCompat.getMainExecutor(this),
        )
    }

    override fun onResume() {
        super.onResume()

        // 앱이 다시 시작될 때마다 권한 요청
        if (!hasPermissions(this)) {
            ActivityCompat.requestPermissions(
                this, permissions.toTypedArray(), permissionsRequestCode,
            )
        } else {
            bindCameraUseCases()
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == permissionsRequestCode && hasPermissions(this)) {
            bindCameraUseCases()
        } else {
            finish() // 필요한 권한이 없는 경우 종료
        }
    }

    /** 앱에서 필요한 모든 권한이 부여되었는지 확인하는 편의 메서드 */
    private fun hasPermissions(context: Context) = permissions.all {
        ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun calculateArmAngle(predictions: List<PoseEstimationHelper.PosePrediction>): ResultData? {
        val prediction = predictions.firstOrNull() ?: return null

        fun calculateAngle(shoulder: PoseEstimationHelper.Position, elbow: PoseEstimationHelper.Position, hip: PoseEstimationHelper.Position): Float {
            val vectorSE = PoseEstimationHelper.Position(elbow.x - shoulder.x, elbow.y - shoulder.y)
            val vectorSH = PoseEstimationHelper.Position(hip.x - shoulder.x, hip.y - shoulder.y)

            val dotProduct = vectorSE.x * vectorSH.x + vectorSE.y * vectorSH.y
            val magnitudeSE = Math.sqrt((vectorSE.x * vectorSE.x + vectorSE.y * vectorSE.y).toDouble())
            val magnitudeSH = Math.sqrt((vectorSH.x * vectorSH.x + vectorSH.y * vectorSH.y).toDouble())

            val cosAngle = dotProduct / (magnitudeSE * magnitudeSH)
            return Math.toDegrees(Math.acos(cosAngle.coerceIn(-1.0, 1.0))).toFloat()
        }

        val leftShoulder = prediction.keypoints.find { it.bodyPart == PoseEstimationHelper.BodyPart.LEFT_SHOULDER }?.position
        val leftElbow = prediction.keypoints.find { it.bodyPart == PoseEstimationHelper.BodyPart.LEFT_ELBOW }?.position
        val leftHip = prediction.keypoints.find { it.bodyPart == PoseEstimationHelper.BodyPart.LEFT_HIP }?.position

        val rightShoulder = prediction.keypoints.find { it.bodyPart == PoseEstimationHelper.BodyPart.RIGHT_SHOULDER }?.position
        val rightElbow = prediction.keypoints.find { it.bodyPart == PoseEstimationHelper.BodyPart.RIGHT_ELBOW }?.position
        val rightHip = prediction.keypoints.find { it.bodyPart == PoseEstimationHelper.BodyPart.RIGHT_HIP }?.position

        val leftAngle = if (leftElbow != null && leftShoulder != null && leftHip != null) {
            calculateAngle(leftShoulder, leftElbow, leftHip)
        } else null

        val rightAngle = if (rightElbow != null && rightShoulder != null && rightHip != null) {
            calculateAngle(rightShoulder, rightElbow, rightHip)
        } else null

        if (leftAngle != null && rightAngle != null) {
            Timber.tag("ArmAngle").d("Left: $leftAngle, Right: $rightAngle")
            Timber.tag("Keypoints").d("LeftElbow: $leftElbow, LeftShoulder: $leftShoulder, LeftHip: $leftHip")
            Timber.tag("Keypoints").d("RightElbow: $rightElbow, RightShoulder: $rightShoulder, RightHip: $rightHip")

            return ResultData(leftAngle, rightAngle)
        }

        return null
    }

    private fun togglePoseEstimation() {
        isPoseEstimationEnabled = !isPoseEstimationEnabled
        updatePoseEstimationButton()
    }

    private fun updatePoseEstimationButton() {
        binding.btnTogglePoseEstimation.text = if (isPoseEstimationEnabled) "Pose Est. ON" else "Pose Est. OFF"
        binding.btnTogglePoseEstimation.setBackgroundColor(
            if (isPoseEstimationEnabled)
                ContextCompat.getColor(this, android.R.color.holo_green_light)
            else
                ContextCompat.getColor(this, android.R.color.darker_gray),
        )
    }
}
