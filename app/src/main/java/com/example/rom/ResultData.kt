package com.example.rom

import android.os.Parcelable
import kotlinx.parcelize.Parcelize

@Parcelize
data class ResultData(
    val leftAngle: Float = 0f,
    val rightAngle: Float = 0f,
    val imageByteArray: ByteArray? = null,
) : Parcelable {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as ResultData

        if (leftAngle != other.leftAngle) return false
        if (rightAngle != other.rightAngle) return false
        if (imageByteArray != null) {
            if (other.imageByteArray == null) return false
            if (!imageByteArray.contentEquals(other.imageByteArray)) return false
        } else if (other.imageByteArray != null) return false

        return true
    }

    override fun hashCode(): Int {
        var result = leftAngle.hashCode()
        result = 31 * result + rightAngle.hashCode()
        result = 31 * result + (imageByteArray?.contentHashCode() ?: 0)
        return result
    }
}
