import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition
from torch.autograd import Variable
import time
import sys
from torch import nn
from torchvision import models


# -------------------------------------------
# Custom Dataset (Video Frames -> Faces)
# -------------------------------------------
class DeepfakeDataset(Dataset):
    def __init__(self, video_path, transform=None, max_frames=100):
        self.video_path = video_path
        self.transform = transform
        self.frames = self._extract_frames(max_frames)

    def _extract_frames(self, max_frames):
        frames = []
        cap = cv2.VideoCapture(self.video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // max_frames)

        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb)

                for top, right, bottom, left in face_locations:
                    face = rgb[top:bottom, left:right]
                    frames.append(face)
            count += 1
        cap.release()
        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        face = self.frames[idx]
        if self.transform:
            face = self.transform(face)
        return face


# -------------------------------------------
# Deepfake Detection Model (ResNet18)
# -------------------------------------------
class DeepfakeModel(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepfakeModel, self).__init__()
        base_model = models.resnet18(pretrained=True)
        num_ftrs = base_model.fc.in_features
        base_model.fc = nn.Linear(num_ftrs, num_classes)
        self.model = base_model

    def forward(self, x):
        return self.model(x)


# -------------------------------------------
# Inference Function
# -------------------------------------------
def predict_video(video_path, model_path, device="cpu"):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = DeepfakeDataset(video_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    model = DeepfakeModel(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for faces in dataloader:
            faces = faces.to(device)
            outputs = model(faces)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())

    # Majority Voting
    fake_score = np.mean(preds)
    if fake_score > 0.5:
        return "FAKE", fake_score
    else:
        return "REAL", 1 - fake_score
