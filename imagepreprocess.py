import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

def pad_and_cut(img, patch_size):
    h, w = img.shape[:2]
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    padded_img = np.pad(
        img,
        ((0, pad_h), (0, pad_w), (0, 0)) if img.ndim == 3 else ((0, pad_h), (0, pad_w)),
        mode='constant',
        constant_values=0
    )

    new_h, new_w = padded_img.shape[:2]
    patches = []

    for i in range(0, new_h, patch_size):
        for j in range(0, new_w, patch_size):
            patch = padded_img[i:i+patch_size, j:j+patch_size]
            patches.append(patch)

    return patches

def save_all_patches_flat(input_folder, output_folder, patch_size=256):
    os.makedirs(output_folder, exist_ok=True)
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)]

    global_patch_idx = 0

    for img_file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(input_folder, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Failed to load {img_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        patches = pad_and_cut(img, patch_size)

        for patch in patches:
            if np.sum(patch) < 1e5:
                continue
            patch_img = Image.fromarray(patch)
            patch_filename = f"patch_{global_patch_idx:06d}.png"
            patch_img.save(os.path.join(output_folder, patch_filename))
            global_patch_idx += 1

if __name__ == "__main__":
    input_dir = "./data/Flickr2K_sub"         # 原始图像路径
    output_dir = "./data/patches_flat"    # 所有 patch 存这里
    patch_size = 512

    save_all_patches_flat(input_dir, output_dir, patch_size)
