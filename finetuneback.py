
################## 1. Download checkpoints and build models
import os
if os.path.exists('/content/VAR'): os.chdir('/content/VAR')
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from Datapreprocessing import PreCroppedPatchDataset

################## 2. Define some helper functions for zero-shot editing

from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from models.var import AdaLNSelfAttn, sample_with_top_k_top_p_, gumbel_softmax_with_rng


def get_edit_mask(patch_nums: List[int], y0: float, x0: float, y1: float, x1: float, device, inpainting: bool = True) -> torch.Tensor:
    ph, pw = patch_nums[-1], patch_nums[-1]
    edit_mask = torch.zeros(ph, pw, device=device)
    edit_mask[round(y0 * ph):round(y1 * ph), round(x0 * pw):round(x1 * pw)] = 1 # outpainting mode: center would be gt
    if inpainting:
        edit_mask = 1 - edit_mask   # inpainting mode: center would be model pred
    return edit_mask    # a binary mask, 1 for keeping the tokens of the image to be edited; 0 for generating new tokens (by VAR)

# overwrite the function of 'VAR::autoregressive_infer_cfg'
def autoregressive_infer_cfg_with_mask(
    self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
    g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
    more_smooth=False,
    input_img_tokens: Optional[List[torch.Tensor]] = None, edit_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
    """
    only used for inference, on autoregressive mode
    :param B: batch size
    :param label_B: imagenet label; if None, randomly sampled
    :param g_seed: random seed
    :param cfg: classifier-free guidance ratio
    :param top_k: top-k sampling
    :param top_p: top-p sampling
    :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
    :param input_img_tokens: (optional, only for zero-shot edit tasks) tokens of the image to be edited
    :param edit_mask: (optional, only for zero-shot edit tasks) binary mask, 1 for keeping given tokens; 0 for generating new tokens
    :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
    """
    if g_seed is None: rng = None
    else: self.rng.manual_seed(g_seed); rng = self.rng

    if label_B is None:
        label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
    elif isinstance(label_B, int):
        label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)

    sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))

    lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC

    next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]

    cur_L = 0
    f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])

    for b in self.blocks: b.attn.kv_caching(True)
    for si, pn in enumerate(self.patch_nums):   # si: i-th segment
        ratio = si / self.num_stages_minus_1
        # last_L = cur_L
        cur_L += pn*pn
        # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        x = next_token_map
        AdaLNSelfAttn.forward
        for b in self.blocks:
            x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
        logits_BlV = self.get_logits(x, cond_BD)

        t = cfg * ratio
        logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

        idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
        if not more_smooth: # this is the default case
            h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
        else:   # not used when evaluating FID/IS/Precision/Recall
            gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
            h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

        h_BChw = h_BChw.transpose(1, 2).reshape(B, self.Cvae, pn, pn)
        # if edit_mask is not None:
        gt_BChw = self.vae_quant_proxy[0].embedding(input_img_tokens[si]).transpose(1, 2).reshape(B, self.Cvae, pn, pn).detach()
       
        h_BChw = combine_embedding(h_BChw, gt_BChw, ph=pn, pw=pn)   # TODO: =====> please specify the ratio <=====

        f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat.clone(), h_BChw)
        if si != self.num_stages_minus_1:   # prepare for next stage
            next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
            next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
            next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
      
    for b in self.blocks: b.attn.kv_caching(False)
    return (self.vae_proxy[0].fhat_to_img(f_hat) + 1) * 0.5

def combine_embedding(h_BChw: torch.Tensor, gt_BChw: torch.Tensor, ph: int, pw:int) -> torch.Tensor:
    B = h_BChw.shape[0]
    ratio = 0.2
    return h_BChw * ratio + gt_BChw * (1 - ratio)   # TODO: =====> please specify the ratio <=====




# we recommend using imagenet-512-d36 model to do the in-painting & out-painting & class-condition editing task
MODEL_DEPTH = 24    # TODO: =====> please specify MODEL_DEPTH <=====

assert MODEL_DEPTH in {16, 20, 24, 30, 36}


# download checkpoint
hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
vae_ckpt, var_ckpt = './model_path/var/vae_ch160v4096z32.pth', f'./model_path/var/var_d{MODEL_DEPTH}.pth'
if not osp.exists(vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
if not osp.exists(var_ckpt): os.system(f'wget {hf_home}/{var_ckpt}')

# build vae, var
FOR_512_px = None
if FOR_512_px:
    patch_nums = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)
else:
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
print(f'patch_nums: {patch_nums}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vae, var = build_vae_var(
    V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
    device=device, patch_nums=patch_nums,
    num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
)
vae
var.to(device)
# load checkpoints
# vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
# var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)

vae.eval()
for p in vae.parameters():
    p.requires_grad = False  # freeze VAE

var.train()

# ========= Step 2: Optimizer ============
optimizer = torch.optim.AdamW(var.parameters(), lr=1e-4, weight_decay=0.01)

# ========= Step 3: Dataloader ============
dataset = PreCroppedPatchDataset(folder_path='./data/Flickr2K')
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# ========= Step 4: Training Loop ============
num_epochs = 10
loss_fn = torch.nn.L1Loss()


for epoch in range(num_epochs):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for img in pbar:
        hr_img = img.to(device)

        # Step 1: downsample input img (if needed)
        lr_img = F.interpolate(img, size=(128, 128), mode='bicubic')
        lr_img = F.interpolate(lr_img, size=(256, 256), mode='bicubic')
        lr_img = lr_img.to(device)

        # Step 2: convert image to token list
        ms_idxBl = vae.img_to_idxBl(lr_img)  # List of token sequences for each scale

        # Step 3: generate embeddings autoregressively (simulate inference)
        B = hr_img.shape[0]
        label_B = torch.full((B,), 1000, device=device, dtype=torch.long) 
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            # similar to autoregressive_infer_cfg
            pred_imgs = autoregressive_infer_cfg_with_mask(
                var,
                B=B, label_B=label_B, cfg=3, top_k=900, top_p=0.95, g_seed=0, more_smooth=False,
                input_img_tokens=ms_idxBl
            )

            # Step 4: calculate loss
            loss = loss_fn(pred_imgs, hr_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        pbar.set_postfix({"loss": loss.item()})
        

# ========= Optional: Save ============
os.makedirs('checkpoints', exist_ok=True)
torch.save(var.state_dict(), 'checkpoints/var_finetuned.pth')
