import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

        self.model.load_state_dict(
            torch.load("artifacts/training/trained_resnet18.pth", map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.class_names = [str(i) for i in range(10)]

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        return {
            "prediction": self.class_names[predicted.item()],
            "confidence": round(confidence.item(), 4)
        }
