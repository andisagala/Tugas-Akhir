package com.example.facedetector.MainScreenUI

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import com.example.facedetector.ui.theme.FacedetectorTheme

@Composable
fun MainScreenUI(label: String = "Tak Ada", confidence: String = "0.0") {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Transparent)
    ) {
        Box(
            modifier = Modifier
                .background(Color.Transparent)
                .fillMaxWidth()
                .align(Alignment.BottomCenter)
                .padding(bottom = 100.dp), // <- move this Box to bottom center
            contentAlignment = Alignment.Center
        ) {
            Box(modifier = Modifier.padding(bottom = 0.dp)) {
                Box(
                    modifier = Modifier
                        .background(Color(0xFF474747).copy(alpha = 0.4f))

                ) {
                    Text(
                        text = if (label != "tidak_ada") label else "",
                        fontSize = 50.sp,
                        textAlign = TextAlign.Center,
                        color = Color.White
                    )
                }
            }
        }
    }
}


@Preview(
    showBackground = true,
    device = "spec:width=2400dp," +
            "height=1080dp," +
            "dpi=409," +
            "isRound=false," +
            "chinSize=0dp," +
            "orientation=portrait"
)
@Composable
fun GreetingPreview() {
    FacedetectorTheme {
//        Box(modifier = Modifier.fillMaxSize()) {
//
//            Text(
//                text = "Tempat detail dari \n \n \nperforma aplikasi dan" +
//                        "\n\n \n model",
//                color = Color.White,
//                fontSize = 70.sp,
//                textAlign = TextAlign.Center,
//                modifier = Modifier
//                    .background(Color.Black.copy(alpha = 0.5f))
//                    .padding(top = 0.dp)
//                    .align(Alignment.TopCenter)
//                    .fillMaxWidth()
//            )
//            }

        MainScreenUI(label = "Teks Terjemahan")

//        Box(
//            modifier = Modifier.fillMaxSize(),
//            contentAlignment = Alignment.Center // Centers the content inside the Box
//        ) {
//            Text(
//                text = "Gambar muncul disini",
//                color = Color.White,
//                fontSize = 80.sp,
//                textAlign = TextAlign.Center,
//                modifier = Modifier
//                    .background(Color.Black.copy(alpha = 0.5f))
//                    .fillMaxWidth()
//            )
//        }
    }
}