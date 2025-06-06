# File for calculating features from ResNet50
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn as nn
import glob
from tqdm import tqdm
import numpy as np
import re

class ImageFolderDataset(Dataset):
    # Assumes that the path is the folder of one video, inside just appear .jpg files
    def __init__(self, video_folder):
        self.video_folder = video_folder
        self.image_files = glob.glob(os.path.join(video_folder, "*.jpg")) + glob.glob(os.path.join(video_folder, "*.png"))
        # Transforms tomadas del codigo de Transvnet
        self.transform = transforms.Compose([
                            transforms.Resize((250, 250)),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]
                            )
                        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx] 



if __name__ == "__main__":

    # Model configurations
    resnet50 = models.resnet50(pretrained=True)
    resnet50.eval()
    feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])  # Quitar capa fc

    # Usar GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor.to(device)

    # Folders for LED videos
    base_path_dir = "DATASETS/PHASES/frames"
    # List of all the paths with the videos
    video_folders = sorted(glob.glob(os.path.join(base_path_dir, "*")), key=lambda x: int(re.search(r'\d+$', x).group()))

    video_folders = video_folders[101:]

    for video_folder in tqdm(video_folders):
        print(f"Processing video folder: {video_folder}")
        dataset = ImageFolderDataset(video_folder)
        dataloader = DataLoader(dataset, batch_size=2048, shuffle=False)
        feature_list = []

        with torch.no_grad():
            for images, filenames in tqdm(dataloader):
                images = images.to(device)
                features = feature_extractor(images)  # (batch_size, 2048, 1, 1)
                features = features.view(features.size(0), -1)  # (batch_size, 2048)
                feature_list.append(features.cpu())
        
        # Apilar todo en un solo tensor y convertirlo a numpy
        feature_list = torch.cat(feature_list, dim=0).numpy()  # (total_frames, 2048)
        # Save all the image features in a .npy file
        video_name = os.path.basename(video_folder)
        output_file = os.path.join('Resnet50_video_features', f"{video_name}_features.npy")
        np.save(output_file, feature_list)




