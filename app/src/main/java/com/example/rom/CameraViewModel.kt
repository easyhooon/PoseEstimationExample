package com.example.rom

import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

class CameraViewModel: ViewModel() {
    private val _isPoseEstimationEnabled = MutableStateFlow(false)
    val isPoseEstimationEnabled: StateFlow<Boolean> = _isPoseEstimationEnabled.asStateFlow()

    fun togglePoseEstimation() {
        _isPoseEstimationEnabled.value = !_isPoseEstimationEnabled.value
    }
}
