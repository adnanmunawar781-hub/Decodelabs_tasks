import urllib.request

print("🌐 Downloading MobileNet-SSD architecture (prototxt)...")
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
    "MobileNetSSD.prototxt"
)

print("📦 Downloading MobileNet-SSD weights (caffemodel)...")
print("⏳ Please wait, this is around 22MB and might take a minute depending on internet speed...")
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/mobilenet_iter_73000.caffemodel",
    "MobileNetSSD.caffemodel"
)

print("✅ Download Complete! Both files are successfully saved in your folder.")