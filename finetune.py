import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from Datapreprocessing import PreCroppedPatchDataset
from models import build_vae_var
from torch.amp import autocast

# ==== Config ====
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_DEPTH = 24  # VAR-d24
PATCH_NUMS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
NUM_CLASSES = 1000
NUM_EPOCHS = 10
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01

# ==== Dataset ====
dataset = PreCroppedPatchDataset(folder_path='./data/Flickr2K')
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==== Model: VAE + VAR ====
vae_ckpt = './model_path/var/vae_ch160v4096z32.pth'
var_ckpt = f'./model_path/var/var_d{MODEL_DEPTH}.pth'

vae, var = build_vae_var(
    V=4096, Cvae=32, ch=160, share_quant_resi=4,
    device=device, patch_nums=PATCH_NUMS,
    num_classes=NUM_CLASSES, depth=MODEL_DEPTH, shared_aln=False,
)

# Load weights
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)

vae.eval()
for p in vae.parameters():
    p.requires_grad = False
vae.to(device)
var.train().to(device)

# ==== Optimizer ====
optimizer = torch.optim.AdamW(var.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# ==== Training Loop ====
for epoch in range(5):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for img in pbar:
        img = img.to(device)

        # === Step 1: 低清图像处理（可跳过/自定义）===
        lr_img = F.interpolate(img, size=(128, 128), mode='bicubic')
        lr_img = F.interpolate(lr_img, size=(256, 256), mode='bicubic')

        # === Step 2: 获取全 scale token ===
        gt_idxBl = vae.img_to_idxBl(lr_img)                              # list of [B, Ls]
        target_tokens = torch.cat(gt_idxBl, dim=1)                       # [B, L]
        input_embeds = vae.quantize.idxBl_to_var_input(gt_idxBl)        # [B, L-no-sos, Cvae]

        label_B = torch.full((img.size(0),), 1000, device=device, dtype=torch.long)

        # === Step 3: Forward + Loss ===
        with autocast(device_type='cuda', dtype=torch.float16):
            logits = var(label_B, input_embeds)  # [B, L, V]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_tokens.view(-1))

        # === Step 4: Backprop ===
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()


        pbar.set_postfix({"loss": loss.item()})


# ==== Save Fine-Tuned Model ====
os.makedirs('checkpoints', exist_ok=True)
torch.save(var.state_dict(), 'checkpoints/var_finetuned_tokenloss.pth')
print("✅ Model saved to checkpoints/var_finetuned_tokenloss.pth")
