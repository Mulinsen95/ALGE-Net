import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    简单的卷积块：Conv -> BN -> ReLU
    """
    def __init__(self, in_ch, out_ch, norm=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        ]
        if norm:
            layers.insert(1, nn.GroupNorm(num_groups=8, num_channels=out_ch))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class EdgeBranch(nn.Module):
    """
    边缘分支：
      输入: MRI 图像 (B, 1, H, W)
      输出:
        - edge_map: (B, 1, H, W) 经过 Sigmoid 的边缘概率图
        - feats: dict，多尺度特征用于后面和主干融合
        - edge_logits: (B, 1, H, W) 未过 Sigmoid，方便算 BCEWithLogitsLoss
    """
    def __init__(self, in_ch: int = 1, base_ch: int = 32):
        super().__init__()

        # ===== Encoder =====
        self.enc1 = ConvBlock(in_ch, base_ch)            # H,   W
        self.down1 = nn.MaxPool2d(2)                     # H/2, W/2

        self.enc2 = ConvBlock(base_ch, base_ch * 2)      # H/2, W/2
        self.down2 = nn.MaxPool2d(2)                     # H/4, W/4

        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)  # H/4, W/4
        self.down3 = nn.MaxPool2d(2)                     # H/8, W/8

        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)

        # ===== Decoder (U-Net 结构) =====
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4,
                                      kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2,
                                      kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch,
                                      kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        # 最后一层 1x1 conv 输出单通道 edge logits
        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        # ----- Encoder -----
        e1 = self.enc1(x)        # (B, base_ch, H,   W)
        p1 = self.down1(e1)      # (B, base_ch, H/2, W/2)

        e2 = self.enc2(p1)       # (B, 2*base_ch, H/2, W/2)
        p2 = self.down2(e2)      # (B, 2*base_ch, H/4, W/4)

        e3 = self.enc3(p2)       # (B, 4*base_ch, H/4, W/4)
        p3 = self.down3(e3)      # (B, 4*base_ch, H/8, W/8)

        b = self.bottleneck(p3)  # (B, 8*base_ch, H/8, W/8)

        # ----- Decoder -----
        u3 = self.up3(b)         # (B, 4*base_ch, H/4, W/4)
        u3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(u3)       # (B, 4*base_ch, H/4, W/4)

        u2 = self.up2(d3)        # (B, 2*base_ch, H/2, W/2)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)       # (B, 2*base_ch, H/2, W/2)

        u1 = self.up1(d2)        # (B, base_ch, H, W)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)       # (B, base_ch, H, W)

        edge_logits = self.out_conv(d1)     # (B, 1, H, W)
        edge_map = torch.sigmoid(edge_logits)

        # 多尺度特征：后面做融合/一致性约束会用到
        feats = {
            "low": e1,           # 较低层，包含细节边缘(96×96）
            "mid": e2,           # 中层，区域与局部结构（48×48）
            "mid2": e3,          # 次中层，区域与局部结构（24x24）  
            "high": b            # 高层，语义结构（12×12）
        }

        return edge_map, feats, edge_logits


class DiceLoss(nn.Module):
    """
    soft Dice loss，输入为 logits（会内部做 sigmoid）
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (B, 1, H, W)
        # targets: (B, 1, H, W)，0~1
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(1)
        dice = (2. * intersection + self.smooth) / (
            probs_flat.sum(1) + targets_flat.sum(1) + self.smooth
        )
        return 1.0 - dice.mean()


bce_loss_fn = nn.BCEWithLogitsLoss()
dice_loss_fn = DiceLoss()

def edge_loss(edge_logits, edge_target, alpha=1.0, beta=1.0):
    edge_target = edge_target.float()

    # 动态计算 pos_weight = neg/pos
    with torch.no_grad():
        pos = edge_target.sum()
        neg = edge_target.numel() - pos
        pw = (neg / (pos + 1e-6)).clamp(1.0, 50.0)  # 防止过大
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    loss_bce = bce(edge_logits, edge_target)
    loss_dice = dice_loss_fn(edge_logits, edge_target)
    return alpha * loss_bce + beta * loss_dice

