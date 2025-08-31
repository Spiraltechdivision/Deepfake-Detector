from flask import Flask, request, render_template_string
import asyncio
from model_infer import detect_deepfake

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Deepfake Detector</title>
</head>
<body>
    <h2>Upload Video for Deepfake Detection</h2>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="video" required>
        <input type="submit" value="Analyze">
    </form>
    {% if result %}
        <h3>Prediction: {{ result['prediction'] }}</h3>
        <p>Confidence: {{ result['confidence'] }}%</p>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return "No video uploaded", 400

    video_file = request.files["video"]
    save_path = "./uploaded_video.mp4"
    video_file.save(save_path)

    result = asyncio.run(detect_deepfake(save_path))
    result["prediction"] = "REAL" if result["label"] == 0 else "FAKE"

    return render_template_string(HTML_TEMPLATE, result=result)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
