import os, random, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')
    print(f'VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    # ── image & watermark ────────────────────────────────────────────────────────
IMG_SIZE    = 128        # 128×128 keeps VRAM well under 16 GB; use 256 if desired
WATERMARK_BITS = 64      # binary copyright message length

# ── data ─────────────────────────────────────────────────────────────────────
DATA_DIR = "/kaggle/input/datasets/soumikrakshit/div2k-high-resolution-images/DIV2K_train_HR/DIV2K_train_HR"
VAL_DIR = "/kaggle/input/datasets/soumikrakshit/div2k-high-resolution-images/DIV2K_valid_HR/DIV2K_valid_HR"
NUM_TRAIN   = 800        # DIV2K train set size
NUM_VAL     = 100        # DIV2K val set size

# ── training ─────────────────────────────────────────────────────────────────
BATCH_SIZE  = 8          # safe for T4 at 128×128; lower to 4 if OOM
EPOCHS      = 40
LR          = 1e-4
GRAD_CLIP   = 1.0

# ── loss weights (λ1 image, λ2 watermark, λ3 perceptual placeholder) ─────────
LAMBDA_IMG  = 1
LAMBDA_WM   = 1.5       # slightly upweight watermark recovery
LAMBDA_PERC = 0.05        # set > 0 after initial training stabilises

# ── model dims (compact Transformer) ────────────────────────────────────────
EMBED_DIM   = 64         # patch embedding channels
PATCH_SIZE  = 8          # 8×8 patches → 16×16 grid for 128-img
NUM_HEADS   = 4
DEPTH       = 4          # transformer blocks per stage
MLP_RATIO   = 2.0

# ── output dir ───────────────────────────────────────────────────────────────
OUT_DIR = '/kaggle/working/outputs'
os.makedirs(OUT_DIR, exist_ok=True)
print('Config loaded.')

class WatermarkDataset(Dataset):
    """Returns (image_tensor, random_binary_watermark) pairs."""

    def __init__(self, img_dir, size=IMG_SIZE, limit=None, augment=False):
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        self.paths = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if os.path.splitext(f)[1].lower() in exts
        ])
        if limit:
            self.paths = self.paths[:limit]

        aug_list = []
        if augment:
            aug_list += [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            ]
        else:
            aug_list.append(transforms.CenterCrop(size))

        self.transform = transforms.Compose([
            transforms.Resize(size + 16),   # a little extra for crop headroom
            *aug_list,
            transforms.Resize(size),        # ensure exact size
            transforms.ToTensor(),           # [0, 1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        img = self.transform(img)
        wm  = torch.randint(0, 2, (WATERMARK_BITS,)).float()  # random binary message
        return img, wm


train_ds = WatermarkDataset(DATA_DIR, limit=NUM_TRAIN, augment=True)
val_ds   = WatermarkDataset(VAL_DIR,  limit=NUM_VAL,   augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f'Train: {len(train_ds)} images · Val: {len(val_ds)} images')
print(f'Train batches: {len(train_loader)} · Val batches: {len(val_loader)}')

class PatchEmbed(nn.Module):
    """Split image into non-overlapping patches and project to embed_dim."""

    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_ch=3, embed_dim=EMBED_DIM):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)  →  (B, N, embed_dim)
        x = self.proj(x)               # (B, embed_dim, H/P, W/P)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x, H, W


class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer block with GELU MLP."""

    def __init__(self, dim, num_heads, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden     = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Self-attention with residual
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class PatchUnembed(nn.Module):
    """Reshape token sequence back to spatial feature map."""

    def __init__(self, embed_dim=EMBED_DIM, patch_size=PATCH_SIZE, out_ch=3):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, out_ch,
                               kernel_size=patch_size, stride=patch_size),
        )

    def forward(self, x, H, W):
        # x: (B, N, C) → (B, C, H*P, W*P)
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return self.proj(x)  # (B, out_ch, img_size, img_size)

class WatermarkEmbedder(nn.Module):
    """
    Takes (image, watermark_bits) and returns watermarked_image.

    1. Patch-embed the image  →  token sequence
    2. Expand & inject the watermark bits as a learned offset
    3. Run Transformer blocks
    4. Reconstruct the spatial image via PatchUnembed
    5. Add residual from original image for imperceptibility
    """

    def __init__(self):
        super().__init__()
        self.patch_embed   = PatchEmbed()
        num_patches        = self.patch_embed.num_patches

        # Project watermark bits to per-token injection vector
        self.wm_proj = nn.Sequential(
            nn.Linear(WATERMARK_BITS, EMBED_DIM),
            nn.GELU(),
            nn.Linear(EMBED_DIM, EMBED_DIM),
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, EMBED_DIM))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks      = nn.ModuleList([
            TransformerBlock(EMBED_DIM, NUM_HEADS, MLP_RATIO) for _ in range(DEPTH)
        ])
        self.norm        = nn.LayerNorm(EMBED_DIM)
        self.patch_unembed = PatchUnembed(out_ch=3)
        self.out_act     = nn.Tanh()  # small residual in [-1, 1]
        self.residual_scale  = nn.Parameter(torch.tensor(0.1))  # ← ADD THIS


    def forward(self, img, wm):
        # img: (B, 3, H, W) · wm: (B, bits)
        tokens, H, W = self.patch_embed(img)        # (B, N, C)
        tokens = tokens + self.pos_embed

        # Broadcast watermark to every token
        wm_vec = self.wm_proj(wm).unsqueeze(1)      # (B, 1, C)
        tokens = tokens + wm_vec

        for blk in self.blocks:
            tokens = tokens + wm_vec              # re-inject before every block
            tokens = blk(tokens)

        residual    = self.out_act(self.patch_unembed(tokens, H, W))
        # In WatermarkEmbedder.forward():
        watermarked = torch.clamp(img + self.residual_scale.clamp(0.02, 0.12) * residual, 0.0, 1.0)
        return watermarked


print('Embedder defined.')

class WatermarkExtractor(nn.Module):
    """
    Takes a (possibly attacked) watermarked image and predicts the binary
    watermark bits.

    1. Patch-embed the image  →  token sequence
    2. Run Transformer blocks
    3. Global-average-pool the tokens
    4. MLP head  →  logits over WATERMARK_BITS
    """

    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed()
        num_patches      = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, EMBED_DIM))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(EMBED_DIM, NUM_HEADS, MLP_RATIO) for _ in range(DEPTH)
        ])
        self.norm = nn.LayerNorm(EMBED_DIM)

        self.head = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(EMBED_DIM * 2, WATERMARK_BITS),
        )

    def forward(self, img):
        tokens, _, _ = self.patch_embed(img)   # (B, N, C)
        tokens = tokens + self.pos_embed

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        pooled = tokens.mean(dim=1)            # (B, C)  global average pool
        logits = self.head(pooled)             # (B, bits)  — raw logits
        return logits


print('Extractor defined.')


class AttackLayer(nn.Module):
    """
    Randomly applies one of: Gaussian noise · JPEG-approx blur ·
    resize-down-up · random crop-pad · identity.
    All ops are differentiable (or treated as pass-through for gradients).
    """

    def __init__(self, p=0.8):
        super().__init__()
        self.p = p  # probability of applying any attack

    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x

        attack = random.choice(['noise', 'blur', 'resize', 'crop', 'identity'])

        if attack == 'noise':
            sigma = random.uniform(0.01, 0.05)
            x = torch.clamp(x + sigma * torch.randn_like(x), 0, 1)

        elif attack == 'blur':
            # Approximate JPEG artifact / Gaussian blur via avg pool + upsample
            factor = random.choice([2, 4])
            H, W   = x.shape[-2:]
            x = F.avg_pool2d(x, factor)
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        elif attack == 'resize':
            H, W   = x.shape[-2:]
            scale  = random.uniform(0.5, 0.9)
            nh, nw = max(16, int(H * scale)), max(16, int(W * scale))
            x = F.interpolate(x, size=(nh, nw), mode='bilinear', align_corners=False)
            x = F.interpolate(x, size=(H, W),   mode='bilinear', align_corners=False)

        elif attack == 'crop':
            H, W   = x.shape[-2:]
            crop   = random.uniform(0.1, 0.25)
            ch, cw = int(H * crop), int(W * crop)
            # Zero out a random border strip
            x = x.clone()
            x[:, :, :ch, :]  = 0.5
            x[:, :, -ch:, :] = 0.5
            x[:, :, :, :cw]  = 0.5
            x[:, :, :, -cw:] = 0.5

        # identity: no change
        return x


print('AttackLayer defined.')
embedder  = WatermarkEmbedder().to(DEVICE)
extractor = WatermarkExtractor().to(DEVICE)
attack    = AttackLayer(p=0.8).to(DEVICE)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Embedder  params : {count_params(embedder):,}')
print(f'Extractor params : {count_params(extractor):,}')
print(f'Total params     : {count_params(embedder) + count_params(extractor):,}')

# ── loss ─────────────────────────────────────────────────────────────────────
mse_loss = nn.MSELoss()
bce_loss = nn.BCEWithLogitsLoss()

# REPLACE your total_loss function in Section 4:

def total_loss(orig, watermarked, wm_true, wm_logits):
    l_img = mse_loss(watermarked, orig)
    l_wm  = bce_loss(wm_logits, wm_true)

    # Normalise: target img_loss ~ 0.001, target wm_loss ~ 0.693
    # Scale both to same magnitude before applying lambdas
    l_img_scaled = l_img * 1000          # now sits in ~1.0 range
    l_wm_scaled  = l_wm  / 0.693        # now sits in ~1.0 range (0=perfect, 1=random)

    loss = LAMBDA_IMG * l_img_scaled + LAMBDA_WM * l_wm_scaled
    return loss, l_img.item(), l_wm.item()


# ── metrics ──────────────────────────────────────────────────────────────────
def psnr(orig, recon):
    mse = mse_loss(recon, orig).item()
    if mse == 0:
        return float('inf')
    return 10 * math.log10(1.0 / mse)

def ssim_approx(orig, recon):
    """Lightweight SSIM approximation (no external lib needed)."""
    mu1, mu2   = orig.mean(), recon.mean()
    sig1, sig2 = orig.std(),  recon.std()
    sig12      = ((orig - mu1) * (recon - mu2)).mean()
    C1, C2     = (0.01 ** 2), (0.03 ** 2)
    num   = (2 * mu1 * mu2 + C1) * (2 * sig12 + C2)
    denom = (mu1 ** 2 + mu2 ** 2 + C1) * (sig1 ** 2 + sig2 ** 2 + C2)
    return (num / denom).item()

def ber(wm_true, wm_logits):
    """Bit Error Rate: fraction of bits decoded incorrectly."""
    pred = (torch.sigmoid(wm_logits) > 0.5).float()
    return (pred != wm_true).float().mean().item()

def extraction_accuracy(wm_true, wm_logits):
    pred = (torch.sigmoid(wm_logits) > 0.5).float()
    per_sample = (pred == wm_true).float().mean(dim=1)  # per-message accuracy
    return per_sample.mean().item()

print('Loss functions and metrics defined.')

# UPDATED Section 5 — remove optimizer, keep scheduler + scaler:
params    = list(embedder.parameters()) + list(extractor.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                torch.optim.AdamW(params, lr=LR, weight_decay=1e-4),  # dummy for scheduler init
                T_max=EPOCHS, eta_min=1e-6)
scaler    = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

print(f'Optimizer  : AdamW  lr={LR}  wd=1e-4')
print(f'Scheduler  : CosineAnnealingLR  T_max={EPOCHS}')
print(f'AMP (FP16) : {DEVICE.type == "cuda"}')

history = {'train_loss': [], 'val_loss': [], 'psnr': [],
           'ssim': [], 'ber': [], 'acc': []}


def train_epoch(epoch):
    embedder.train(); extractor.train(); attack.train()
    total, img_l, wm_l = 0.0, 0.0, 0.0

    for imgs, wms in train_loader:
        imgs, wms = imgs.to(DEVICE), wms.to(DEVICE)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
            watermarked = embedder(imgs, wms)
            attacked    = attack(watermarked)
            wm_logits   = extractor(attacked)
            loss, li, lw = total_loss(imgs, watermarked, wms, wm_logits)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total += loss.item()
        img_l += li
        wm_l  += lw

    n = len(train_loader)
    return total / n, img_l / n, wm_l / n


@torch.no_grad()
def val_epoch():
    embedder.eval(); extractor.eval(); attack.eval()
    v_loss, v_psnr, v_ssim, v_ber, v_acc = 0.0, 0.0, 0.0, 0.0, 0.0

    for imgs, wms in val_loader:
        imgs, wms = imgs.to(DEVICE), wms.to(DEVICE)

        with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
            watermarked = embedder(imgs, wms)
            wm_logits   = extractor(watermarked)   # no attack at val time
            loss, _, _  = total_loss(imgs, watermarked, wms, wm_logits)

        v_loss += loss.item()
        v_psnr += psnr(imgs, watermarked)
        v_ssim += ssim_approx(imgs, watermarked)
        v_ber  += ber(wms, wm_logits)
        v_acc  += extraction_accuracy(wms, wm_logits)

    n = len(val_loader)
    return v_loss/n, v_psnr/n, v_ssim/n, v_ber/n, v_acc/n


print('Train/val loops defined.')

# =======================================================
# =======================================================
# Training 
# =======================================================
# =======================================================

PHASE1_EPOCHS = 8
best_val_loss = float('inf')

for epoch in range(1, EPOCHS + 1):

    # ── Phase control ────────────────────────────────────────────
    if epoch <= PHASE1_EPOCHS:
        for p in embedder.parameters():
            p.requires_grad = True
        for p in extractor.parameters():
            p.requires_grad = False
        if epoch == 1:
            print('>>> Phase 1: embedder-only training (extractor frozen)')
    else:
        for p in embedder.parameters():
            p.requires_grad = True
        for p in extractor.parameters():
            p.requires_grad = True
        if epoch == PHASE1_EPOCHS + 1:
            print('>>> Phase 2: joint training (all unfrozen)')

    # ── Rebuild optimizer ────────────────────────────────────────
    if epoch <= PHASE1_EPOCHS:
        active_params = list(embedder.parameters())
    else:
        active_params = list(embedder.parameters()) + list(extractor.parameters())
    optimizer = torch.optim.AdamW(active_params, lr=LR, weight_decay=1e-4)

    # ── Train & validate ─────────────────────────────────────────
    t0 = time.time()
    tr_loss, tr_img, tr_wm = train_epoch(epoch)
    vl_loss, vl_psnr, vl_ssim, vl_ber, vl_acc = val_epoch()
    scheduler.step()

    history['train_loss'].append(tr_loss)
    history['val_loss'].append(vl_loss)
    history['psnr'].append(vl_psnr)
    history['ssim'].append(vl_ssim)
    history['ber'].append(vl_ber)
    history['acc'].append(vl_acc)

    elapsed = time.time() - t0
    lr_now  = scheduler.get_last_lr()[0]

    print(f'Epoch {epoch:03d}/{EPOCHS}  '
          f'tr={tr_loss:.4f} (img={tr_img:.4f} wm={tr_wm:.4f})  '
          f'val={vl_loss:.4f}  PSNR={vl_psnr:.2f} dB  '
          f'SSIM={vl_ssim:.4f}  BER={vl_ber:.4f}  '
          f'Acc={vl_acc:.4f}  LR={lr_now:.2e}  {elapsed:.1f}s')

    if vl_loss < best_val_loss:
        best_val_loss = vl_loss
        torch.save({'embedder':  embedder.state_dict(),
                    'extractor': extractor.state_dict(),
                    'epoch':     epoch},
                   f'{OUT_DIR}/best_model.pth')
        print(f'  ↳ best model saved (val_loss={vl_loss:.4f})')

print('\nTraining complete.')
print(f'Accuracy = {(1 - vl_ber) * 100:.2f}%')

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

def plot(ax, data, title, ylabel, color='steelblue'):
    ax.plot(data, color=color)
    ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)

plot(axes[0,0], history['train_loss'], 'Train Loss', 'Loss')
plot(axes[0,1], history['val_loss'],   'Val Loss',   'Loss', color='orange')
plot(axes[0,2], history['psnr'],       'PSNR (dB)',  'dB',   color='green')
plot(axes[1,0], history['ssim'],       'SSIM',       'SSIM', color='purple')
plot(axes[1,1], history['ber'],        'BER',        'BER',  color='red')
plot(axes[1,2], history['acc'],        'Extraction Accuracy', 'Accuracy', color='teal')

plt.suptitle('Training Metrics — Transformer Watermarking', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/training_curves.png', dpi=150)
plt.show()

# Load best checkpoint
ckpt = torch.load(f'{OUT_DIR}/best_model.pth', map_location=DEVICE)
embedder.load_state_dict(ckpt['embedder'])
extractor.load_state_dict(ckpt['extractor'])
embedder.eval(); extractor.eval()


def apply_attack(imgs, attack_name, **kw):
    """Non-differentiable attacks for eval."""
    if attack_name == 'none':
        return imgs
    elif attack_name == 'jpeg':
        # Approx JPEG: downsample + upsample
        H, W = imgs.shape[-2:]
        x = F.avg_pool2d(imgs, 4)
        return F.interpolate(x, (H, W), mode='bilinear', align_corners=False)
    elif attack_name == 'noise':
        sigma = kw.get('sigma', 0.05)
        return torch.clamp(imgs + sigma * torch.randn_like(imgs), 0, 1)
    elif attack_name == 'resize':
        H, W = imgs.shape[-2:]
        x = F.interpolate(imgs, scale_factor=0.5, mode='bilinear', align_corners=False)
        return F.interpolate(x, (H, W), mode='bilinear', align_corners=False)
    elif attack_name == 'crop':
        x = imgs.clone()
        H, W = x.shape[-2:]
        c = H // 5
        x[:, :, :c, :] = 0.5; x[:, :, -c:, :] = 0.5
        x[:, :, :, :c] = 0.5; x[:, :, :, -c:] = 0.5
        return x
    elif attack_name == 'blur':
        H, W = imgs.shape[-2:]
        x = F.avg_pool2d(imgs, 2)
        return F.interpolate(x, (H, W), mode='bilinear', align_corners=False)
    return imgs


ATTACK_NAMES = ['none', 'jpeg', 'noise', 'resize', 'crop', 'blur']
ATTACK_LABELS = ['No attack', 'JPEG compression', 'Gaussian noise',
                 'Resize', 'Crop', 'Blur']

results = {name: {'psnr': 0, 'ssim': 0, 'ber': 0, 'acc': 0}
           for name in ATTACK_NAMES}

with torch.no_grad():
    for imgs, wms in val_loader:
        imgs, wms = imgs.to(DEVICE), wms.to(DEVICE)
        watermarked = embedder(imgs, wms)

        for atk in ATTACK_NAMES:
            attacked  = apply_attack(watermarked, atk)
            wm_logits = extractor(attacked)
            results[atk]['psnr'] += psnr(imgs, watermarked)
            results[atk]['ssim'] += ssim_approx(imgs, watermarked)
            results[atk]['ber']  += ber(wms, wm_logits)
            results[atk]['acc']  += extraction_accuracy(wms, wm_logits)

n = len(val_loader)
for atk in ATTACK_NAMES:
    for k in results[atk]:
        results[atk][k] /= n

print('\nAttack Robustness Results')
print(f'{"Attack":<22} {"PSNR (dB)":>10} {"SSIM":>8} {"BER":>8} {"Acc":>8}')
print('-' * 62)
for atk, label in zip(ATTACK_NAMES, ATTACK_LABELS):
    r = results[atk]
    print(f'{label:<22} {r["psnr"]:>10.2f} {r["ssim"]:>8.4f} '
          f'{r["ber"]:>8.4f} {r["acc"]:>8.4f}')

    def visualise_demo(n_samples=4):
    embedder.eval(); extractor.eval()
    imgs, wms = next(iter(val_loader))
    imgs, wms = imgs[:n_samples].to(DEVICE), wms[:n_samples].to(DEVICE)

    with torch.no_grad():
        watermarked = embedder(imgs, wms)
        diff        = (watermarked - imgs).abs() * 10   # amplified for visibility
        attacked    = apply_attack(watermarked, 'jpeg')
        logits      = extractor(attacked)
        pred_bits   = (torch.sigmoid(logits) > 0.5).float()

    def t(x): return x.cpu().permute(1, 2, 0).clamp(0, 1).numpy()

    fig, axes = plt.subplots(n_samples, 5, figsize=(18, 4 * n_samples))
    cols = ['Original', 'Watermarked', 'Diff (×10)', 'Attacked', 'Bit match']

    for i in range(n_samples):
        b_match = (pred_bits[i] == wms[i]).float().mean().item()
        for j, (img, title) in enumerate(zip(
            [imgs[i], watermarked[i], diff[i], attacked[i]], cols[:4])):
            axes[i, j].imshow(t(img))
            axes[i, j].set_title(title if i == 0 else '')
            axes[i, j].axis('off')
        # Bit accuracy panel
        axes[i, 4].bar(['BIT ACC'], [b_match], color='steelblue')
        axes[i, 4].set_ylim(0, 1)
        axes[i, 4].set_title(cols[4] if i == 0 else '')
        axes[i, 4].set_ylabel(f'{b_match:.2%}')

    plt.suptitle('Demo: Original → Watermarked → Attacked → Extracted',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/demo_visual.png', dpi=150)
    plt.show()


visualise_demo()
print('Demo saved.')      

torch.save({'embedder':  embedder.state_dict(),
            'extractor': extractor.state_dict(),
            'history':   history,
            'config': {
                'IMG_SIZE': IMG_SIZE, 'WATERMARK_BITS': WATERMARK_BITS,
                'EMBED_DIM': EMBED_DIM, 'PATCH_SIZE': PATCH_SIZE,
                'NUM_HEADS': NUM_HEADS, 'DEPTH': DEPTH,
            }},
           f'{OUT_DIR}/final_model.pth')

print('\n=== Final model saved ===')
print(f'Outputs at : {OUT_DIR}')
print('Files      :', os.listdir(OUT_DIR))

best = min(range(EPOCHS), key=lambda i: history['val_loss'][i])
print(f'\nBest epoch : {best + 1}')
print(f'  PSNR     : {history["psnr"][best]:.2f} dB  (target > 30 dB)')
print(f'  SSIM     : {history["ssim"][best]:.4f}     (target > 0.90)')
print(f'  BER      : {history["ber"][best]:.4f}      (target < 0.10)')
print(f'  Acc      : {history["acc"][best]:.4f}      (target > 0.90)')