import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from src.cnnClassifier.entity.config_entity import TrainingConfig
from torchvision import models



class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        #  DATA TRANSFORM
        transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor()
        ])

        #  DATASETS
        train_dataset = datasets.ImageFolder(
            root=self.config.training_data,  
            transform=transform
        )


        #  DATALOADERS
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

    
        #  MODEL (load prepared ResNet18)
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 10) 

        state_dict = torch.load(self.config.base_model_path)
        model.load_state_dict(state_dict)

        model = model.to(self.device)

        #  LOSS & OPTIMIZER
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)

        #  TRAINING LOOP
        for epoch in range(self.config.epochs):
            model.train()
            total_loss = 0
           
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
           

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{self.config.epochs}] Loss: {avg_loss:.4f}")

        #  SAVE TRAINED MODEL
        torch.save(model.state_dict(), self.config.trained_model_path)

      
        