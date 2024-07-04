import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import java.nio.ByteBuffer
import java.nio.ByteOrder

class PoseEstimationHelper(private val tflite: Interpreter) {

    companion object {
        const val IMAGE_SIZE = 192
        const val NUM_KEY_POINTS = 17
    }

    data class PosePrediction(val keypoints: List<KeyPoint>, val score: Float)
    data class KeyPoint(val bodyPart: BodyPart, val position: Position, val score: Float)
    data class Position(val y: Float, val x: Float)

    enum class BodyPart {
        LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
        LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
        LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    }

    fun predict(tensorImage: TensorImage): List<PosePrediction> {
        val inputBuffer = tensorImage.buffer

        // 출력 텐서의 shape을 가져옵니다.
        val outputShape = tflite.getOutputTensor(0).shape()
        val outputSize = outputShape.fold(1) { acc, i -> acc * i }

        val outputBuffer = ByteBuffer.allocateDirect(outputSize * 4) // 4 bytes per float
        outputBuffer.order(ByteOrder.nativeOrder())

        tflite.run(inputBuffer, outputBuffer)

        outputBuffer.rewind()
        val outputs = outputBuffer.asFloatBuffer()

        val keyPoints = mutableListOf<KeyPoint>()
        for (i in 0 until NUM_KEY_POINTS) {
            if (outputs.remaining() >= 3) {
                val y = outputs.get()
                val x = outputs.get()
                val score = outputs.get()
                // 얼굴 부위(처음 5개)를 제외하고 키포인트를 추가합니다.
                if (i >= 5) {
                    keyPoints.add(KeyPoint(BodyPart.entries[i - 5], Position(y, x), score))
                }
            } else {
                break
            }
        }

        val averageScore = keyPoints.map { it.score }.average().toFloat()
        return listOf(PosePrediction(keyPoints, averageScore))
    }
}
