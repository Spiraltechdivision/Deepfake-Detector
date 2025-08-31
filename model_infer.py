import torch
from torch import nn
from torchvision import models, transforms
from torch.utils.data import Dataset
import cv2
import numpy as np
import os

# -------------------
# Model Definition
# -------------------
class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(DeepfakeDetector, self).__init__()
        base_model = models.resnext50_32x4d(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(2048, num_classes)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.feature_extractor(x)
        x = self.global_pool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dropout(self.classifier(x_lstm[:, -1, :]))

# -------------------
# Video Dataset Loader
# -------------------
class VideoDataset(Dataset):
    def __init__(self, video_paths, sequence_length=20, transform=None):
        self.video_paths = video_paths
        self.transform = transform
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frames = []
        for _, frame in enumerate(self.extract_frames(video_path)):
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
            if len(frames) == self.sequence_length:
                break

        frames = torch.stack(frames)
        frames = frames[:self.sequence_length]
        return frames.unsqueeze(0)

    def extract_frames(self, path):
        cap = cv2.VideoCapture(path)
        success, image = cap.read()
        while success:
            yield image
            success, image = cap.read()
        cap.release()

# -------------------
# Utility Functions
# -------------------
softmax = nn.Softmax()

async def classify_video_frames(model, frames_tensor):
    """Run forward pass and return prediction + confidence."""
    _, logits = model(frames_tensor.to("cpu"))
    probs = softmax(logits)
    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0, predicted_class].item() * 100

    return {
        "label": predicted_class,
        "confidence": confidence
    }

async def detect_deepfake(video_path: str, model_path="./model_97_acc_100_frames_FF_data.pt"):
    """Full pipeline: load model, process video, return result."""
    im_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    dataset = VideoDataset([video_path], sequence_length=20, transform=transform)

    model = DeepfakeDetector(num_classes=2)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    result = await classify_video_frames(model, dataset[0])
    return result
