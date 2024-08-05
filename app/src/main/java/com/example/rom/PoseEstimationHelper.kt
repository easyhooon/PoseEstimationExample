import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import java.nio.ByteBuffer
import java.nio.ByteOrder

class PoseEstimationHelper(private val tflite: Interpreter) {

    companion object {
        const val IMAGE_SIZE = 192 // 입력 이미지 크기
        const val NUM_KEY_POINTS = 17 // 전체 키포인트 수 (얼굴 포함)
    }

    // 포즈 예측 결과를 나타내는 데이터 클래스
    data class PosePrediction(val keypoints: List<KeyPoint>, val score: Float)
    // 각 키포인트의 정보를 나타내는 데이터 클래스
    data class KeyPoint(val bodyPart: BodyPart, val position: Position, val score: Float)
    // 키포인트의 2D 위치를 나타내는 데이터 클래스
    data class Position(val y: Float, val x: Float)

    // 신체 부위를 나타내는 열거형 (얼굴 부위 제외)
    enum class BodyPart {
        NOSE,
        LEFT_EYE,
        RIGHT_EYE,
        LEFT_EAR,
        RIGHT_EAR,
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_ELBOW,
        RIGHT_ELBOW,
        LEFT_WRIST,
        RIGHT_WRIST,
        LEFT_HIP,
        RIGHT_HIP,
        LEFT_KNEE,
        RIGHT_KNEE,
        LEFT_ANKLE,
        RIGHT_ANKLE,
    }

    // TensorFlow Lite 모델을 사용하여 포즈 예측을 수행하는 함수
    fun predict(tensorImage: TensorImage): List<PosePrediction> {
        val inputBuffer = tensorImage.buffer

        // 출력 텐서의 shape을 가져와 총 크기를 계산
        val outputShape = tflite.getOutputTensor(0).shape()
        val outputSize = outputShape.fold(1) { acc, i -> acc * i }

        // 출력을 저장할 ByteBuffer 생성
        val outputBuffer = ByteBuffer.allocateDirect(outputSize * 4) // 4 bytes per float
        outputBuffer.order(ByteOrder.nativeOrder())

        // TensorFlow Lite 모델 실행
        tflite.run(inputBuffer, outputBuffer)

        // 출력 버퍼를 FloatBuffer로 변환
        outputBuffer.rewind()
        val outputs = outputBuffer.asFloatBuffer()

        // 키포인트 추출
        val keyPoints = mutableListOf<KeyPoint>()
        for (i in 0 until NUM_KEY_POINTS) {
            if (outputs.remaining() >= 3) {
                val y = outputs.get()
                val x = outputs.get()
                val score = outputs.get()
                // 얼굴 부위(처음 5개)를 제외하고 키포인트를 추가
//                if (i >= 5) {
//                    keyPoints.add(KeyPoint(BodyPart.entries[i - 5], Position(y, x), score))
//                }
                keyPoints.add(KeyPoint(BodyPart.entries[i], Position(y, x), score))
            } else {
                break
            }
        }

        // 전체 키포인트의 평균 점수 계산
        val averageScore = keyPoints.map { it.score }.average().toFloat()

        // PosePrediction 객체를 리스트로 반환
        return listOf(PosePrediction(keyPoints, averageScore))
    }
}

