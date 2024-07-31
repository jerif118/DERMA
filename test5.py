import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import ResNet101_Weights
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

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
        y_label = torch.tensor(1 if self.annotations.iloc[idx]['benign_malignant'] == 'malignant' else 0, dtype=torch.float)

        if self.transform:
            image = self.transform(image)

        return image, y_label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def create_model():
    model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
    
    for param in model.parameters():
        param.requires_grad = False
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 2)
    )
    
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20, device='cuda'):
    best_model_wts = None
    best_acc = 0.0
    patience = 5
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs = inputs.to(device)
                labels = labels.to(device).long()

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        val_auc = roc_auc_score(val_labels, val_preds)

        print(f'Validation - Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} AUC: {val_auc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print("Early stopping")
            break

        scheduler.step(val_acc)
        current_lr = scheduler.get_last_lr()[0]
        print(f'Current Learning Rate: {current_lr}')

    if best_model_wts:
        model.load_state_dict(best_model_wts)
    return model

def main():
    train_csv_file = './combined_train_metadata.csv'
    root_dir = './Augmented-ISIC-images'
    
    full_dataset = SkinDataset(csv_file=train_csv_file, root_dir=root_dir, transform=transform)
    
    # Balanceo de clases usando WeightedRandomSampler
    class_counts = full_dataset.annotations['benign_malignant'].value_counts()
    class_weights = 1. / class_counts
    sample_weights = full_dataset.annotations['benign_malignant'].map(class_weights).values
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_sampler = WeightedRandomSampler(sample_weights[train_dataset.indices], len(train_dataset.indices))
    val_sampler = WeightedRandomSampler(sample_weights[val_dataset.indices], len(val_dataset.indices))
    
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, sampler=val_sampler, num_workers=4, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    model = create_model()
    model.to(device)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
    
    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30, device=device)
    
    torch.save(model.state_dict(), './model/derma.pth')

if __name__ == '__main__':
    main()
