import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path


class Evaluation:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   

   
    def get_validation_loader(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        dataset = datasets.ImageFolder(
            root=self.config.training_data,
            transform=transform
        )

        test_loader = DataLoader(
            dataset,
            batch_size=self.config.params_batch_size,
            shuffle=False
        )

        return test_loader
    
    def load_model(self):
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, self.config.num_classes)

        model.load_state_dict(torch.load(self.config.path_of_model, map_location=self.device))
        model.to(self.device)
        model.eval()

        return model
    
    def evaluate(self):
        test_loader = self.get_validation_loader() 
        model = self.load_model()
        loss_fn = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                loss = loss_fn(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total

        print(f"✅ Test Loss     : {avg_loss:.4f}")
        print(f"✅ Test Accuracy : {accuracy * 100:.2f}%")

        self.score = {"loss": avg_loss, "accuracy": accuracy}

  
    def save_score(self):
        with open("scores.json", "w") as f:
            f.write(str(self.score))



    

    