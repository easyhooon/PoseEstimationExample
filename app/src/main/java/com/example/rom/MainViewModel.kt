package com.example.rom

import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

class MainViewModel: ViewModel() {
    private val _resultData = MutableStateFlow(ResultData())
    val resultData: StateFlow<ResultData> = _resultData.asStateFlow()

    private val _validationMessage = MutableStateFlow("")
    val validationMessage: StateFlow<String> = _validationMessage.asStateFlow()

    fun setResultData(resultData: ResultData) {
        _resultData.value = resultData
    }

    fun setValidationMessage(validationMessage: String) {
        _validationMessage.value = validationMessage
    }
}
