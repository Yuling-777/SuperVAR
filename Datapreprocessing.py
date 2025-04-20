from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import cv2
import numpy as np
from utils.data import pil_loader, normalize_01_into_pm1

class ImagePatchDataset(Dataset):
    def __init__(self, folder_path, patch_size=256, device='cpu'):
        self.folder_path = folder_path
        self.patch_size = patch_size
        self.device = device
        self.patches = self._load_and_patch_images()

        # 预处理器：PIL -> tensor [C, H, W] -> [-1, 1]
        self.transform = transforms.Compose([
            transforms.ToTensor(),            # [0, 255] -> [0, 1]
            normalize_01_into_pm1             # [0, 1] -> [-1, 1]
        ])

    def _pad_and_cut(self, img):
        h, w = img.shape[:2]
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size

        padded_img = np.pad(
            img,
            ((0, pad_h), (0, pad_w), (0, 0)) if img.ndim == 3 else ((0, pad_h), (0, pad_w)),
            mode='constant',
            constant_values=0
        )

        new_h, new_w = padded_img.shape[:2]
        patches = []

        for i in range(0, new_h, self.patch_size):
            for j in range(0, new_w, self.patch_size):
                patch = padded_img[i:i+self.patch_size, j:j+self.patch_size]
                patches.append(patch)
        
        return patches

    def _load_and_patch_images(self):
        patches = []
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(image_extensions)]

        for img_file in image_files:
            img_path = os.path.join(self.folder_path, img_file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_patches = self._pad_and_cut(img)
            patches.extend(img_patches)

        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch_np = self.patches[idx]
        patch_pil = Image.fromarray(patch_np)
        patch_tensor = self.transform(patch_pil).to(self.device)  # apply ToTensor + normalize
        return patch_tensor



class PreCroppedPatchDataset(Dataset):
    def __init__(self, folder_path, device='cpu'):
        self.folder_path = folder_path
        self.device = device
        self.image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        self.transform = transforms.Compose([
            transforms.CenterCrop(512),
            transforms.ToTensor(),       # [0, 255] → [0, 1]
            normalize_01_into_pm1        # [0, 1] → [-1, 1]
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB')
        tensor = self.transform(img).to(self.device)
        return tensor