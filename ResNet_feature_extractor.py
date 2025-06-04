# File for calculating features from ResNet50

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn as nn

# -------------------------
# 1. Definir transformaciones
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(  # Normalización para ImageNet
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------
# 2. Dataset personalizado
# -------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]  # También devolvemos el nombre del archivo

# -------------------------
# 3. DataLoader
# -------------------------
image_dir = "/ruta/a/tu/carpeta_de_imagenes"  # <-- cambia esto
dataset = ImageFolderDataset(image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# -------------------------
# 4. Modelo ResNet50 sin la capa final
# -------------------------
resnet50 = models.resnet50(pretrained=True)
resnet50.eval()
feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])  # Quitar capa fc

# Usar GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor.to(device)

# -------------------------
# 5. Extraer features por lotes
# -------------------------
all_features = []
all_filenames = []

with torch.no_grad():
    for images, filenames in dataloader:
        images = images.to(device)
        features = feature_extractor(images)  # (batch_size, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # (batch_size, 2048)
        all_features.append(features.cpu())
        all_filenames.extend(filenames)

# Concatenar todos los features
features_tensor = torch.cat(all_features, dim=0)
print("Features shape:", features_tensor.shape)



