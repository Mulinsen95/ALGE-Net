import torch
import torch.nn as nn
import torch.nn.functional as F

from models.KANResnet import resnet50
from models.Edge_branch import EdgeBranch, edge_loss
from models.KAN import KAN


# =========================================================
# 基础组件：必要结构以匹配权重文件
# =========================================================
class SpatialAttention(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, feat):
        return torch.sigmoid(self.conv(feat))


class SkullAwareSobelGuide(nn.Module):
    def __init__(self, tau: float = 0.4):
        super().__init__()
        self.tau = tau
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, x):
        gx = F.conv2d(x, self.kernel_x, padding=1)
        gy = F.conv2d(x, self.kernel_y, padding=1)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

        B = mag.shape[0]
        flat = mag.view(B, -1)
        vmax = torch.quantile(flat, 0.98, dim=1).view(B, 1, 1, 1)
        vmin = torch.quantile(flat, 0.02, dim=1).view(B, 1, 1, 1)
        mag = torch.clamp((mag - vmin) / (vmax - vmin + 1e-8), 0, 1)

        edge = (mag > self.tau).float()
        for _ in range(2):
            edge = (F.avg_pool2d(edge, 3, stride=1, padding=1) > 0.5).float()
        edge = F.max_pool2d(edge, 3, stride=1, padding=1)
        return F.avg_pool2d(edge, 3, stride=1, padding=1)


class EdgeFusionBlock(nn.Module):
    def __init__(self, age_ch: int, edge_ch: int):
        super().__init__()
        self.edge_to_age = nn.Conv2d(edge_ch, age_ch, kernel_size=1)
        self.age_bias_fc = nn.Linear(age_ch, age_ch)
        nn.init.zeros_(self.age_bias_fc.weight)
        nn.init.zeros_(self.age_bias_fc.bias)

    def forward(self, age_feat, edge_feat):
        # 1. 空间对齐与通道转换
        edge_resized = F.interpolate(edge_feat, size=age_feat.shape[-2:], mode='bilinear', align_corners=False)
        edge_adapt = self.edge_to_age(edge_resized)

        # 2. 生成门控信号
        age_ctx = F.adaptive_avg_pool2d(age_feat, 1).flatten(1)
        age_bias = self.age_bias_fc(age_ctx).unsqueeze(-1).unsqueeze(-1)
        gate = torch.tanh(edge_adapt + age_bias)

        # 3. 融合并池化输出
        fused = age_feat * (1.0 + gate)
        pooled = F.adaptive_avg_pool2d(fused, 1).flatten(1)
        return fused, pooled


# =========================================================
# 主模型：双分支融合
# =========================================================
class DualBranchModel(nn.Module):
    def __init__(self, small_input=True, dropout=None, fusion_mode="all",
                 lambda_edge=0.5, lambda_cons=0.1, lambda_sobel=0.2):
        super().__init__()
        assert fusion_mode in ["shallow", "middle", "deep", "all"]
        self.fusion_mode = fusion_mode
        self.lambda_edge, self.lambda_cons, self.lambda_sobel = lambda_edge, lambda_cons, lambda_sobel

        # 主干与边缘分支
        self.backbone = resnet50(small_input=small_input, dropout=dropout)
        self.edge_branch = EdgeBranch(in_ch=1, base_ch=32)

        # 获取各层输出通道
        def get_ch(layer): return layer[-1].bn3.num_features if hasattr(layer[-1], "bn3") else layer[
            -1].bn2.num_features

        ch_l = [get_ch(self.backbone.layer1), get_ch(self.backbone.layer2), get_ch(self.backbone.layer3),
                get_ch(self.backbone.layer4)]

        # 融合层初始化 (对应 EdgeBranch 的 64, 128, 256, 256 通道)
        self.fuse_l1 = EdgeFusionBlock(ch_l[0], 64)
        self.fuse_l2 = EdgeFusionBlock(ch_l[1], 128)
        self.fuse_l3 = EdgeFusionBlock(ch_l[2], 256)
        self.fuse_l4 = EdgeFusionBlock(ch_l[3], 256)

        # 回归头 KAN
        in_dim = sum(ch_l) if fusion_mode == "all" else (
            ch_l[2] + ch_l[3] if fusion_mode == "deep" else ch_l[0] + ch_l[1])
        self.reg_head = KAN([in_dim, 64, 1])

        # 辅助模块
        self.spatial_attn = SpatialAttention(ch_l[2])
        self.sobel_guide = SkullAwareSobelGuide(tau=0.4)
        self.age_loss_fn = nn.SmoothL1Loss(beta=1.0)

    def forward(self, x, age_gt=None, edge_target=None):
        # 1. 特征提取
        _, age_feats = self.backbone._forward_backbone(x)
        _, edge_feats, edge_logits = self.edge_branch(x)

        # 2. 逐层融合
        fused_vecs = []
        # Stage 1 & 2
        _, p1 = self.fuse_l1(age_feats["l1"], edge_feats["mid"])
        _, p2 = self.fuse_l2(age_feats["l2"], edge_feats["mid2"])
        # Stage 3 & 4
        _, p3 = self.fuse_l3(age_feats["l3"], edge_feats["high"])
        _, p4 = self.fuse_l4(age_feats["l4"], edge_feats["high"])

        # 3. 拼接与预测
        if self.fusion_mode == "all":
            fused_vec = torch.cat([p1, p2, p3, p4], dim=1)
        elif self.fusion_mode == "deep":
            fused_vec = torch.cat([p3, p4], dim=1)
        else:  # shallow
            fused_vec = torch.cat([p1, p2], dim=1)

        age_pred = self.reg_head(fused_vec)

        outputs = {"age_pred": age_pred}

        # 4. 仅在训练时计算 Loss
        if age_gt is not None:
            age_gt = age_gt.view_as(age_pred).float()
            loss_age = self.age_loss_fn(age_pred, age_gt)

            # 边缘与一致性 Loss
            loss_edge = edge_loss(edge_logits, edge_target) if edge_target is not None else torch.tensor(0.0).to(
                x.device)

            # 计算一致性损失
            loss_cons = F.l1_loss(self.spatial_attn(age_feats["l3"]),
                                  F.interpolate(edge_logits, size=age_feats["l3"].shape[-2:]).detach())

            with torch.no_grad():
                sobel_target = self.sobel_guide(x)
            loss_sobel = F.mse_loss(torch.sigmoid(edge_logits), sobel_target)

            # 计算总 Loss
            outputs[
                "loss_total"] = loss_age + self.lambda_edge * loss_edge + self.lambda_cons * loss_cons + self.lambda_sobel * loss_sobel

            # 这里包含 train.py 需要的所有键
            outputs.update({
                "loss_age": loss_age,
                "loss_edge": loss_edge,
                "loss_cons": loss_cons,
                "loss_sobel": loss_sobel
            })

        return outputs

    # ================= 可视化 =================
    def forward_vis(self, x):
        """专门用于可视化的前向传播，返回中间特征图"""
        # 1. 提取边缘分支结果
        edge_map, _, edge_logits = self.edge_branch(x)

        # 2. 获取 Sobel 物理先验（作为对比基准）
        with torch.no_grad():
            sobel_target = self.sobel_guide(x)

        return {
            "sobel_target": sobel_target,
            "edge_map": edge_map,  # 模型重建后的边缘
            "edge_logits": edge_logits
        }