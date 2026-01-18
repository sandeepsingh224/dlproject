import os
from pathlib import Path
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from src.cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from src.cnnClassifier.utils.common import save_model



class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        # Load pretrained ResNet18
        if self.config.pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        model = models.resnet18(weights=weights)

        # Freeze backbone (industry practice)
        for param in model.parameters():
            param.requires_grad = False

        # Replace classifier head
        model.fc = torch.nn.Linear(
            model.fc.in_features,
            self.config.classes
        )

        self.model = model

        # Save base model
        save_model(
            path=self.config.base_model_path,
            model=self.model
        )
