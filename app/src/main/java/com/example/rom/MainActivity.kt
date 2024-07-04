package com.example.rom

import android.content.Intent
import android.os.Build.VERSION.SDK_INT
import android.os.Build.VERSION_CODES.TIRAMISU
import android.os.Bundle
import android.view.View
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import coil.load
import com.example.rom.databinding.ActivityMainBinding
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {
    companion object {
        const val EXTRA_POSE_ESTIMATION_RESULT = "pose_estimation_result"
    }

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraActivityResultLauncher: ActivityResultLauncher<Intent>

    private val viewModel by viewModels<MainViewModel>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraActivityResultLauncher = registerForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) { result ->
            val resultData: ResultData?
            if (result.resultCode == RESULT_OK) {
                resultData = if (SDK_INT >= TIRAMISU) {
                    result.data?.getParcelableExtra(EXTRA_POSE_ESTIMATION_RESULT, ResultData::class.java)
                } else {
                    result.data?.getParcelableExtra(EXTRA_POSE_ESTIMATION_RESULT)
                }

                viewModel.setResultData(resultData!!)
            }
        }

        binding.btnCapture.setOnClickListener {
            cameraActivityResultLauncher.launch(Intent(this, CameraActivity::class.java))
        }

        lifecycleScope.launch {
            lifecycle.repeatOnLifecycle(Lifecycle.State.STARTED) {
                viewModel.resultData.collect {
                    if (it.leftAngle == 0f && it.rightAngle == 0f && it.imageByteArray == null) {
                        binding.apply {
                            ivResultImage.visibility = View.GONE

                            tvResult.visibility = View.GONE

                            tvLeftAngle.visibility = View.GONE
                            tvLeftAngleValue.visibility = View.GONE

                            tvRightAngle.visibility = View.GONE
                            tvRightAngleValue.visibility = View.GONE
                        }
                    } else {
                        binding.apply {
                            ivResultImage.load(it.imageByteArray)
                            ivResultImage.visibility = View.VISIBLE

                            btnCapture.text = "다시 촬영"
                            tvResult.visibility = View.VISIBLE

                            tvLeftAngle.visibility = View.VISIBLE
                            tvLeftAngleValue.text = it.leftAngle.toString()
                            tvLeftAngleValue.visibility = View.VISIBLE

                            tvRightAngle.visibility = View.VISIBLE
                            tvRightAngleValue.text = it.rightAngle.toString()
                            tvRightAngleValue.visibility = View.VISIBLE
                        }
                    }
                }
            }
        }
    }
}
