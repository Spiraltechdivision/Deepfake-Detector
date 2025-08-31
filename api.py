from flask import Flask, request, jsonify
import asyncio
from model_infer import detect_deepfake

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze_video():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video_file = request.files["video"]
    save_path = "./uploaded_video.mp4"
    video_file.save(save_path)

    result = asyncio.run(detect_deepfake(save_path))
    label = "REAL" if result["label"] == 0 else "FAKE"

    return jsonify({
        "prediction": label,
        "confidence": result["confidence"]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
