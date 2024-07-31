# Importar librerías
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm

# Definir Dataset personalizado
class SkinDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx]['isic_id'] + ".jpg")
        image = Image.open(img_name).convert('RGB')
        y_label = torch.tensor(1 if self.annotations.iloc[idx]['benign_malignant'] == 'malignant' else 0)

        if self.transform:
            image = self.transform(image)

        return image, y_label

# Transformaciones con aumento de datos
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),   # Rotación horizontal
    transforms.RandomRotation(10),       # Rotación aleatoria de hasta 10 grados
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Recorte aleatorio
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Cambios de color
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Definir transformaciones
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def main():
    # Cargar datos y dividir en conjuntos de entrenamiento y prueba
    csv_file = './metadata.csv'
    root_dir = './ISIC-imagest'
    full_dataset = SkinDataset(csv_file=csv_file, root_dir=root_dir, transform=train_transform)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_dataset.dataset.transform = train_transform
    test_dataset.dataset.transform = test_transform

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Definir y configurar el modelo
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),  
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    model.to(device)

    # Definir función de pérdida y optimizador
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # Definir función para calcular la precisión
    def calculate_accuracy(outputs, labels):
        predicted = (outputs > 0.5).float()
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        return correct / total

    # Entrenar el modelo
    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        train_loader_tqdm = tqdm(train_loader, total=len(train_loader))
        for images, labels in train_loader_tqdm:
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_accuracy += calculate_accuracy(outputs, labels)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = running_accuracy / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device).float().view(-1, 1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_accuracy += calculate_accuracy(outputs, labels)
        
        val_loss /= len(test_loader)
        val_accuracy /= len(test_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Val_Loss: {val_loss:.4f}, Val_Accuracy: {val_accuracy:.4f}')

    # Guardar el modelo entrenado
    torch.save(model.state_dict(), './model/deram2.pth')

if __name__ == '__main__':
    main()
