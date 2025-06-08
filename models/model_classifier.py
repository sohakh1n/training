import torch
import torch.nn as nn

# Leichte Datenaugmentation für Spektrogramme
class SpecMasking(nn.Module):
    def __init__(self, freq_mask=24, time_mask=48):        
        super().__init__()
        self.freq_mask = freq_mask
        self.time_mask = time_mask

    def forward(self, x):
        if not self.training:
            return x
        B, C, F, T = x.size()
        for i in range(B):
            f_start = torch.randint(0, F - self.freq_mask, (1,))
            t_start = torch.randint(0, T - self.time_mask, (1,))
            x[i, :, f_start:f_start+self.freq_mask, :] = 0
            x[i, :, :, t_start:t_start+self.time_mask] = 0
        return x

# Ein Block aus Conv → BatchNorm → GELU → (optional Pooling)
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.block(x))

# Attention-Pooling Layer für Feature-Zusammenfassung
class AttentionPool(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        attn_out, _ = self.attn(x, x, x)
        return self.norm(attn_out.mean(dim=1))  # [B, C]

# Finales Audio-Klassifikationsnetz
class AudioCNN(nn.Module):
    def __init__(self, n_mels=128, n_steps=431, n_classes=50):
        super().__init__()
        self.specaug = SpecMasking()
        self.encoder = nn.Sequential(
        ConvBlock(1, 32),
        ConvBlock(32, 64),
        ConvBlock(64, 128),
        ConvBlock(128, 128),         # ← zusätzlicher Layer
        ConvBlock(128, 128, pool=False)  # ← ohne Pooling am Ende

)
        self.attnpool = AttentionPool(128)
        self.fc = nn.Sequential(
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Dropout(0.5),   # ← statt 0.3
        nn.Linear(128, n_classes)
)

    def forward(self, x):
        x = self.specaug(x)
        x = self.encoder(x)
        x = self.attnpool(x)
        return self.fc(x)
