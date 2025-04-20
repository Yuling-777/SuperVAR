import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from Datapreprocessing import PreCroppedPatchDataset
from models import build_vae_var
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter


# ==== Config ====
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_DEPTH = 16  # VAR-d24
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
writer = SummaryWriter(log_dir='runs/var_finetune')

# ==== Optimizer ====
optimizer = torch.optim.AdamW(var.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# ==== Training Loop ====
torch.autograd.set_detect_anomaly(True)

for epoch in range(NUM_EPOCHS):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for img in pbar:
        img = img.to(device)

        # === Step 1: 低清图像处理（可跳过/自定义）===
        lr_img = F.interpolate(img, size=(128, 128), mode='bicubic')
        lr_img = F.interpolate(lr_img, size=(256, 256), mode='bicubic')

        # === Step 2: 获取 VAE 的 multi-scale token 作为 ground-truth ===
        with torch.no_grad():
            gt_idxBl = vae.img_to_idxBl(lr_img)  # list of [B, Ls]
            gt_embeds = [vae.quantize.embedding(x.to(device)) for x in gt_idxBl]  # list of [B, Ls, Cvae]

        # === Step 3: 模拟 autoregressive 逻辑，但使用 gt_embed 替代 f_hat ===
        B = img.size(0)
        label_B = torch.full((B,), 1000, device=device, dtype=torch.long)
        sos = cond_BD = var.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=NUM_CLASSES)), dim=0))
        lvl_pos = var.lvl_embed(var.lvl_1L) + var.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2 * B, var.first_l, -1) + var.pos_start.expand(2 * B, var.first_l, -1) + lvl_pos[:, :var.first_l]
        cond_BD_or_gss = var.shared_ada_lin(cond_BD)

        cur_L = 0
        loss = 0.0

        for b in var.blocks:
            b.attn.kv_caching(True)

        for si, pn in enumerate(PATCH_NUMS):
            ratio = si / (len(PATCH_NUMS) - 1)
            cur_L += pn * pn

            x = next_token_map
            for block in var.blocks:
                x = block(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)

            logits_BlV = var.get_logits(x, cond_BD)
            t = 1.5 * ratio  # classifier-free guidance
            logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

            probs = F.softmax(logits_BlV, dim=-1)  # [B, L, V]
            pred_embed = probs @ vae.quantize.embedding.weight  # [B, L, Cvae]
            h_BChw = pred_embed.transpose(1, 2).reshape(B, var.Cvae, pn, pn)

            
            if si != len(PATCH_NUMS) - 1:
                h_BChw = F.interpolate(h_BChw, size=(PATCH_NUMS[si + 1],PATCH_NUMS[si + 1]), mode='area')
                gt_BChw = gt_embeds[si + 1].transpose(1, 2).reshape(B, var.Cvae, PATCH_NUMS[si + 1],PATCH_NUMS[si + 1])
            else:
                continue

            loss += F.mse_loss(h_BChw, gt_BChw)

            # 使用 ground-truth 的 token 构造下一阶段输入，不使用 f_hat
            next_token_map = gt_BChw
            if si != len(PATCH_NUMS) - 1:
                next_token_map = next_token_map.contiguous().view(B, -1, var.Cvae)
                next_token_map = var.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + PATCH_NUMS[si + 1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)

        for b in var.blocks:
            b.attn.kv_caching(False)

        # === Step 4: Backprop ===
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        pbar.set_postfix({"loss": loss.item()})
        global_step = epoch * len(train_loader) + pbar.n  # pbar.n 表示当前 step 数
        writer.add_scalar('Loss/train', loss.item(), global_step)

writer.close()

# ==== Save Fine-Tuned Model ====
os.makedirs('checkpoints', exist_ok=True)
torch.save(var.state_dict(), 'checkpoints/var_finetuned_tokenloss.pth')
print("✅ Model saved to checkpoints/var_finetuned_tokenloss.pth")
