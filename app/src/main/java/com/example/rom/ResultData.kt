package com.example.rom

import android.os.Parcelable
import kotlinx.parcelize.Parcelize

@Parcelize
data class ResultData(
    val leftAngleBefore: Float = 0f,
    val rightAngleBefore: Float = 0f,
    val leftShoulderAngle: Float = 0f,
    val rightShoulderAngle: Float = 0f,
    val leftElbowAngle: Float = 0f,
    val rightElbowAngle: Float = 0f,
    val leftAngleAfter: Float = 0f,
    val rightAngleAfter: Float = 0f,
    val imageByteArray: ByteArray? = null,
) : Parcelable {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as ResultData

        if (leftAngleBefore != other.leftAngleBefore) return false
        if (rightAngleBefore != other.rightAngleBefore) return false
        if (leftShoulderAngle != other.leftShoulderAngle) return false
        if (rightShoulderAngle != other.rightShoulderAngle) return false
        if (leftElbowAngle != other.leftElbowAngle) return false
        if (rightElbowAngle != other.rightElbowAngle) return false
        if (leftAngleAfter != other.leftAngleAfter) return false
        if (rightAngleAfter != other.rightAngleAfter) return false
        if (imageByteArray != null) {
            if (other.imageByteArray == null) return false
            if (!imageByteArray.contentEquals(other.imageByteArray)) return false
        } else if (other.imageByteArray != null) return false

        return true
    }

    override fun hashCode(): Int {
        var result = leftAngleBefore.hashCode()
        result = 31 * result + rightAngleBefore.hashCode()
        result = 31 * result + leftShoulderAngle.hashCode()
        result = 31 * result + rightShoulderAngle.hashCode()
        result = 31 * result + leftElbowAngle.hashCode()
        result = 31 * result + rightElbowAngle.hashCode()
        result = 31 * result + leftAngleAfter.hashCode()
        result = 31 * result + rightAngleAfter.hashCode()
        result = 31 * result + (imageByteArray?.contentHashCode() ?: 0)
        return result
    }

}
