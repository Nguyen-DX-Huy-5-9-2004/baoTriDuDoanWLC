package com.example.myapplication

import android.annotation.SuppressLint
import android.os.Bundle
import android.view.ViewGroup
import android.webkit.WebChromeClient
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.tv.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.RectangleShape
import androidx.compose.ui.tooling.preview.Preview
import androidx.tv.material3.ExperimentalTvMaterial3Api
import androidx.tv.material3.Surface
import com.example.myapplication.ui.theme.MyApplicationTheme
import androidx.compose.ui.viewinterop.AndroidView

class MainActivity : ComponentActivity() {
    @OptIn(ExperimentalTvMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MyApplicationTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    shape = RectangleShape
                ) {
                    //Greeting("Nguyễn ĐỨc Huy")
                    StreamlitDashboard()
                }
            }
        }
    }
}

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    Text(
        text = "Hello $name!",
        modifier = modifier
    )
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    MyApplicationTheme {
        Greeting("Android")
    }
}
@SuppressLint("SetJavaScriptEnabled")
@Composable
fun StreamlitDashboard() {
    AndroidView(
        factory = { context ->
            WebView(context).apply {
                layoutParams = ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT
                )
                setLayerType(android.view.View.LAYER_TYPE_SOFTWARE, null)

                settings.apply {
                    settings.allowFileAccess = true
                    settings.allowContentAccess = true
                    javaScriptEnabled = true
                    domStorageEnabled = true
                    databaseEnabled = true
                    useWideViewPort = true
                    loadWithOverviewMode = true
                    mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
                    textZoom = 100
                    setSupportZoom(false)
                }

                webViewClient = object : WebViewClient() {
                    override fun onReceivedError(
                        view: WebView?,
                        errorCode: Int,
                        description: String?,
                        failingUrl: String?
                    ) {
                        view?.loadData(
                            "<html><body><h2>Lỗi kết nối: $description</h2></body></html>",
                            "text/html",
                            "UTF-8"
                        )
                    }
                }
                webChromeClient = WebChromeClient()
            }
        },
        modifier = Modifier.fillMaxSize(),
        update = { webView ->
            // Chỉ gọi loadUrl ở block update để tránh nghẽn luồng lúc factory đang dựng UI
            if (webView.url == null) {
                webView.loadUrl("http://10.0.2.2:5173")
            }
        }
    )
}