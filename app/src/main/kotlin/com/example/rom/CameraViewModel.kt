package com.example.rom

import PoseEstimationHelper
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.launch
import timber.log.Timber
import kotlin.math.acos
import kotlin.math.sqrt

class CameraViewModel : ViewModel() {
    private val _isPoseEstimationEnabled = MutableStateFlow(false)
    val isPoseEstimationEnabled: StateFlow<Boolean> = _isPoseEstimationEnabled.asStateFlow()

    private val _remainingSeconds = MutableStateFlow(10)
    val remainingSeconds: StateFlow<Int> get() = _remainingSeconds

    private val _isCountdownActive = MutableStateFlow(false)
    val isCountdownActive: StateFlow<Boolean> = _isCountdownActive

    private var countdownJob: Job? = null

    fun startCountdownTimer() {
        _isCountdownActive.value = true
        countdownJob = viewModelScope.launch {
            (9 downTo 0).asFlow()
                .onCompletion {
                    _isCountdownActive.value = false
                    countdownJob = null
                }
                .collect { remainingSeconds ->
                    delay(1000)
                    _remainingSeconds.value = remainingSeconds
                }
        }
    }

    fun togglePoseEstimation() {
        _isPoseEstimationEnabled.value = !_isPoseEstimationEnabled.value
    }

    fun calculateArmAngle(predictions: List<PoseEstimationHelper.PosePrediction>): Pair<ResultData?, String> {
        val prediction = predictions.firstOrNull() ?: return Pair(null, "포즈를 감지할 수 없습니다.")

        val leftShoulder = prediction.keypoints.find { it.bodyPart == PoseEstimationHelper.BodyPart.LEFT_SHOULDER }?.position
        val leftElbow = prediction.keypoints.find { it.bodyPart == PoseEstimationHelper.BodyPart.LEFT_ELBOW }?.position
        val leftWrist = prediction.keypoints.find { it.bodyPart == PoseEstimationHelper.BodyPart.LEFT_WRIST }?.position
        val leftHip = prediction.keypoints.find { it.bodyPart == PoseEstimationHelper.BodyPart.LEFT_HIP }?.position

        val rightShoulder = prediction.keypoints.find { it.bodyPart == PoseEstimationHelper.BodyPart.RIGHT_SHOULDER }?.position
        val rightElbow = prediction.keypoints.find { it.bodyPart == PoseEstimationHelper.BodyPart.RIGHT_ELBOW }?.position
        val rightWrist = prediction.keypoints.find { it.bodyPart == PoseEstimationHelper.BodyPart.RIGHT_WRIST }?.position
        val rightHip = prediction.keypoints.find { it.bodyPart == PoseEstimationHelper.BodyPart.RIGHT_HIP }?.position

        if (leftShoulder != null && leftElbow != null && leftWrist != null && leftHip != null &&
            rightShoulder != null && rightElbow != null && rightWrist != null && rightHip != null
        ) {

            // 기본 각도 계산
            val leftBaseAngle = calculateAngle(leftHip, leftShoulder, leftElbow)
            val rightBaseAngle = calculateAngle(rightHip, rightShoulder, rightElbow)

            // 어깨 각도 계산
            val leftShoulderAngle = calculateAngle(rightShoulder, leftShoulder, leftHip)
            val rightShoulderAngle = calculateAngle(leftShoulder, rightShoulder, rightHip)

            val leftElbowAngle = calculateAngle(leftShoulder, leftElbow, leftWrist)
            val rightElbowAngle = calculateAngle(rightShoulder, rightElbow, rightWrist)

            // 최종 각도 계산 (보정 적용)
            val leftFinalAngle = leftBaseAngle - (90f - leftShoulderAngle)
            val rightFinalAngle = rightBaseAngle - (90f - rightShoulderAngle)

            Timber.tag("ArmAngle").d("Left Base: $leftBaseAngle, Right Base: $rightBaseAngle")
            Timber.tag("ArmAngle").d("Left Shoulder: $leftShoulderAngle, Right Shoulder: $rightShoulderAngle")
            Timber.tag("ArmAngle").d("Left Elbow: $leftElbowAngle, Right Elbow: $rightElbowAngle")
            Timber.tag("ArmAngle").d("Left Final: $leftFinalAngle, Right Final: $rightFinalAngle")

            val (isValid, message) = validatePose(prediction)

            // ResultData에 최종 각도를 전달
            return Pair(
                ResultData(
                    leftBaseAngle,
                    rightBaseAngle,
                    leftShoulderAngle,
                    rightShoulderAngle,
                    leftElbowAngle,
                    rightElbowAngle,
                    leftFinalAngle,
                    rightFinalAngle,
                ),
                if (isValid) "" else message,
            )
        }

        return Pair(null, "각도를 계산할 수 없습니다.")
    }

    private fun validatePose(prediction: PoseEstimationHelper.PosePrediction): Pair<Boolean, String> {
        val leftShoulder = prediction.keypoints.find { it.bodyPart == PoseEstimationHelper.BodyPart.LEFT_SHOULDER }?.position
        val leftElbow = prediction.keypoints.find { it.bodyPart == PoseEstimationHelper.BodyPart.LEFT_ELBOW }?.position
        val leftWrist = prediction.keypoints.find { it.bodyPart == PoseEstimationHelper.BodyPart.LEFT_WRIST }?.position

        val rightShoulder = prediction.keypoints.find { it.bodyPart == PoseEstimationHelper.BodyPart.RIGHT_SHOULDER }?.position
        val rightElbow = prediction.keypoints.find { it.bodyPart == PoseEstimationHelper.BodyPart.RIGHT_ELBOW }?.position
        val rightWrist = prediction.keypoints.find { it.bodyPart == PoseEstimationHelper.BodyPart.RIGHT_WRIST }?.position

        // 왼쪽 팔꿈치 각도 검사
        if (leftShoulder != null && leftElbow != null && leftWrist != null) {
            val leftElbowAngle = calculateAngle(leftShoulder, leftElbow, leftWrist)
            Timber.tag("validatePose").d("Left Elbow: $leftElbowAngle")
            if (leftElbowAngle < 170) {
                return Pair(false, "왼쪽 팔꿈치가 10도 이상 접혀있습니다. 다시 촬영해주세요.")
            }
        }

        // 오른쪽 팔꿈치 각도 검사
        if (rightShoulder != null && rightElbow != null && rightWrist != null) {
            val rightElbowAngle = calculateAngle(rightShoulder, rightElbow, rightWrist)
            Timber.tag("validatePose").d("Right Elbow: $rightElbowAngle")
            if (rightElbowAngle < 170) {
                return Pair(false, "오른쪽 팔꿈치가 10도 이상 접혀있습니다. 다시 촬영해주세요.")
            }
        }

//        // 어깨 기울기 검사
//        if (leftShoulder != null && rightShoulder != null) {
//            val shoulderSlope = calculateShoulderSlope(leftShoulder, rightShoulder)
//            val slopeInDegrees = Math.toDegrees(atan(shoulderSlope.toDouble())).toFloat()
//            if (abs(slopeInDegrees) > 5) {
//                val higherShoulder = if (shoulderSlope > 0) "오른쪽" else "왼쪽"
//                return Pair(false, "양쪽 어깨의 높이가 5도 이상 차이납니다. $higherShoulder 어깨가 ${String.format("%.1f", abs(slopeInDegrees))}도 더 높습니다. 어깨를 수평으로 맞추고 다시 촬영해주세요.")
//            }
//        }

        return Pair(true, "")
    }

    private fun calculateAngle(
        point1: PoseEstimationHelper.Position,
        point2: PoseEstimationHelper.Position,
        point3: PoseEstimationHelper.Position,
    ): Float {
        // point2 를 기준점으로 하여 point1 과 point3 로의 벡터를 생성
        val vector1 = PoseEstimationHelper.Position(point1.x - point2.x, point1.y - point2.y)
        val vector2 = PoseEstimationHelper.Position(point3.x - point2.x, point3.y - point2.y)

        // 두 벡터의 내적을 계산
        val dotProduct = vector1.x * vector2.x + vector1.y * vector2.y
        // 두 벡터의 크기를 계산
        val magnitude1 = sqrt((vector1.x * vector1.x + vector1.y * vector1.y).toDouble())
        val magnitude2 = sqrt((vector2.x * vector2.x + vector2.y * vector2.y).toDouble())

        // 코사인 각도 계산
        val cosAngle = dotProduct / (magnitude1 * magnitude2)

        // 아크 코사인 계산 및 라디안에서 도(degree)로 변환
        // 1. cosAngle의 아크코사인을 구하여 라디안 각도를 얻음.
        // 2. coerceIn(-1.0, 1.0)은 코사인 값을 -1에서 1 사이로 제한 (부동소수점 오차 방지)
        // 3. Math.toDegrees()를 사용하여 라디안을 도로 변환
        return Math.toDegrees(acos(cosAngle.coerceIn(-1.0, 1.0))).toFloat()
    }

    fun calculateShoulderSlope(leftShoulder: PoseEstimationHelper.Position, rightShoulder: PoseEstimationHelper.Position): Float {
        val deltaY = rightShoulder.y - leftShoulder.y
        val deltaX = rightShoulder.x - leftShoulder.x
        // 기울기 계산 (y 좌표가 위로 갈수록 작아지므로 -를 붙임)
        return -(deltaY / deltaX)
    }
}
