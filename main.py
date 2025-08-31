import asyncio
import sys
from model_infer import detect_deepfake

async def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_path>")
        return
    
    video_path = sys.argv[1]
    print(f"Analyzing video: {video_path}")

    result = await detect_deepfake(video_path)
    label = "REAL" if result["label"] == 0 else "FAKE"

    print(f"Prediction: {label}")
    print(f"Confidence: {result['confidence']:.2f}%")

if __name__ == "__main__":
    asyncio.run(main())
