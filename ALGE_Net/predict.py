import os
import argparse
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

import os
import sys
from pathlib import Path

# 添加项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.utils1 import (
    evaluateR2_week_seaborn_sequence,
    set_seed,
    read_csv_random_split_data,
    group_by_sequence,
    plot_overall_figures
)
from data.my_dataset_seq import MyDataSetSeq
from models.Dual_branch_model import DualBranchModel

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =========================================================
# Wrapper：确保只返回年龄预测值，保持接口纯净
# =========================================================
class DualModelWrapper(torch.nn.Module):
    def __init__(self, dual_model):
        super().__init__()
        self.dual_model = dual_model

    def forward(self, x):
        out1 = self.dual_model(x)["age_pred"]
        x_flipped = torch.flip(x, dims=[3])
        out2 = self.dual_model(x_flipped)["age_pred"]
        return (out1 + out2) / 2.0


# =========================================================
# 可视化工具函数
# =========================================================
def collect_vis_candidates(dual_model, test_loader, device):
    """
    在单个 seed 的测试集中收集可视化候选样本,使用“边缘锐度(方差)”作为挑选标准。
    """
    dual_model.eval()
    candidates = []
    seen_seqs = set()

    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device).view(1, 1, 3, 3)

    for val_images, val_labels, seq_ids in test_loader:
        for img_tensor, sid in zip(val_images, seq_ids):
            if sid in seen_seqs:
                continue

            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(device)

            with torch.no_grad():
                _, _, edge_logits = dual_model.edge_branch(img_tensor)
                edge_map = torch.sigmoid(edge_logits)
                edge_map = edge_map ** 2.5

                gx = F.conv2d(img_tensor, sobel_x, padding=1)
                gy = F.conv2d(img_tensor, sobel_y, padding=1)
                sobel_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

                s_min, s_max = sobel_mag.min(), sobel_mag.max()
                sobel_target = (sobel_mag - s_min) / (s_max - s_min + 1e-8)

            quality_score = torch.var(edge_map).item()

            candidates.append({
                "img": img_tensor.squeeze().cpu().numpy(),
                "sobel": sobel_target.squeeze().cpu().numpy(),
                "edge": edge_map.squeeze().cpu().numpy(),
                "score": quality_score,  # 使用锐度得分
                "seq_id": sid
            })
            seen_seqs.add(sid)

    return candidates


def plot_global_best_samples(
        selected_samples,
        output_dir="results",
        base_name="figure_5_flat_revise",
        dpi=600,
):

    num_samples = len(selected_samples)
    if num_samples == 0:
        print("[Figure 5] 没有可视化样本，跳过导出。")
        return

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(nrows=4, ncols=num_samples, figsize=(8.0, 5.5))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    row_titles = [
        "Input MRI",
        "Sobel Target",
        "Reconstructed Edge",
        "Edge Response Visualization",
    ]

    mean, std = 0.326930, 0.129822

    for col in range(num_samples):
        sample = selected_samples[col]

        input_vis = np.clip(
            (sample["img"] * std + mean) * 255, 0, 255
        ).astype(np.uint8)
        sobel_vis = np.clip(sample["sobel"] * 255, 0, 255).astype(np.uint8)
        edge_vis = np.clip(sample["edge"] * 255, 0, 255).astype(np.uint8)

        edge_response_vis = cv2.applyColorMap(edge_vis, cv2.COLORMAP_JET)
        edge_response_vis_rgb = cv2.cvtColor(edge_response_vis, cv2.COLOR_BGR2RGB)

        images = [input_vis, sobel_vis, edge_vis, edge_response_vis_rgb]

        for row in range(4):
            ax = axes[row, col]
            ax.imshow(images[row], cmap="gray" if row < 3 else None)
            ax.axis("off")

            if col == 0:
                ax.text(
                    -0.15,
                    0.5,
                    row_titles[row],
                    va="center",
                    ha="right",
                    transform=ax.transAxes,
                    fontsize=10,
                    fontweight="bold",
                    color="#2D3B45",
                )

    png_path = os.path.join(output_dir, f"{base_name}.png")
    tif_path = os.path.join(output_dir, "Figure_5_revise.tif")
    pdf_path = os.path.join(output_dir, f"{base_name}.pdf")

    
    fig.savefig(
        png_path,
        dpi=dpi,
        format="png",
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
        transparent=False,
    )

    with Image.open(png_path) as image:
        image.convert("RGB").save(
            tif_path,
            format="TIFF",
            compression="tiff_lzw",
            dpi=(dpi, dpi),
        )

    fig.savefig(
        pdf_path,
        dpi=dpi,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
        transparent=False,
    )

    plt.close(fig)

    print("\n[Figure 5] 已导出：")
    print(f"  PNG（Overleaf）: {os.path.abspath(png_path)}")
    print(f"  TIFF（投稿附件）: {os.path.abspath(tif_path)}")
    print(f"  PDF（本地备份）: {os.path.abspath(pdf_path)}")


# =========================================================
# 主逻辑：预测与评测
# =========================================================
def main(args):
    # 服务器优先使用 GPU；也可通过 --device cpu / cuda:0 手动指定。
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    df = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}.csv"))
    _, test_images_path = read_csv_random_split_data(df)
    grouped_data = group_by_sequence(test_images_path, data_dir=args.data_dir)

    data_transform = {
        "test": transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.326930], [0.129822])
        ])
    }

    test_dataset = MyDataSetSeq(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        images_path=test_images_path,
        seq_df=grouped_data,
        transform=data_transform["test"]
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=MyDataSetSeq.collater()
    )

    dual_model = DualBranchModel(
        small_input=True,
        dropout=None,
        fusion_mode="all"
    ).to(device)

    weight_dir = args.weight_dir
    weight_name = f"model-best-{args.seed2}-1.pth"
    model_weight_path = os.path.join(weight_dir, weight_name)

    if os.path.exists(model_weight_path):
        dual_model.load_state_dict(
            torch.load(model_weight_path, map_location=device),
            strict=False
        )
    else:
        print(f"⚠️ Warning: Missing weights at {model_weight_path}")

    model = DualModelWrapper(dual_model).to(device)
    model.eval()

    # ========== 收集可视化候选 ==========
    candidates = []
    if args.visualize:
        candidates = collect_vis_candidates(dual_model, test_loader, device)

    # 接收返回的 x_w (真实值) 和 y_w (预测值)
    with torch.no_grad():
        R2, MAE, RMSE, MAPE, GM, x_w, y_w = evaluateR2_week_seaborn_sequence(
            model, test_loader, device,
            args.batch_size, seed=args.seed2
        )

    print(f"Seed {args.seed2} | MAE={MAE:.3f} | R²={R2:.3f}")
    return R2, MAE, RMSE, MAPE, GM, x_w, y_w, candidates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_dir", type=str, default="D://Yulia//KANResnet//HSResNet_base//fetal_data")
    parser.add_argument("--dataset", type=str, default="newdataWindowU1")
    parser.add_argument(
        "--weight_dir",
        type=str,
        default="D://Yulia//KANResNet//HSResNet_base//ED-KANet//weights_dual",
        help="包含 model-best-{seed}-1.pth 的目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Figure 5 输出目录",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto、cpu、cuda 或 cuda:0",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10",
        help="例如 1 或 1,2,3 或 1,2,...,10",
    )
    parser.add_argument("--seed2", type=int, default=None)
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--visualize", action="store_true", default=True, help="生成测试图像的可视化对比图")

    args = parser.parse_args()

    start_time = time.time()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]

    all_results = []

    # 用来存储所有 seed 的预测结果，用于画总图
    global_x_w = []
    global_y_w = []
    global_candidates = []  

    print("\n" + "=" * 50)
    print(f"{'Seed':<8} | {'MAE':<8} | {'R2':<8} | {'RMSE':<8} | {'MAPE':<8} | {'GM':<8}")
    print("-" * 50)

    for seed in seeds:
        set_seed(seed)
        args.seed2 = seed

        r2, mae, rmse, mape, gm, x_w, y_w, cands = main(args)

        global_x_w.extend(x_w.tolist())
        global_y_w.extend(y_w.tolist())
        global_candidates.extend(cands)  

        all_results.append({
            "Seed": seed,
            "MAE": mae,
            "R2": r2,
            "RMSE": rmse,
            "MAPE": mape,
            "GM": gm
        })

        print(f"{seed:<8} | {mae:<8.4f} | {r2:<8.4f} | {rmse:<8.4f} | {mape:<8.4f} | {gm:<8.4f}")

    
    if args.visualize and len(global_candidates) > 0:

        unique_cands = {}
        for c in global_candidates:
            sid = c["seq_id"]
            if sid not in unique_cands or c["score"] > unique_cands[sid]["score"]:
                unique_cands[sid] = c

        sorted_cands = sorted(unique_cands.values(), key=lambda x: x["score"], reverse=True)

        best_3 = sorted_cands[:3]

        # 为了选出“有一定伪影但还行”的样本，我们从列表的 40%-60% 区间随机抽取 3 张
        # 这样不会选到完全是黑块的废图，而是选到有噪点、边缘模糊的真实表现。
        mid_start = int(len(sorted_cands) * 0.4)
        mid_end = int(len(sorted_cands) * 0.8)

        import random

        artifact_cands = sorted_cands[mid_start:mid_end]
        random_3_mediocre = random.sample(artifact_cands, min(3, len(artifact_cands)))

        final_selection = best_3 + random_3_mediocre

        os.makedirs("results", exist_ok=True)
        plot_global_best_samples(
            final_selection,
            output_dir=args.output_dir,
            base_name="figure_5_flat_revise",
            dpi=600,
        )

    # --- 统计与汇总 ---
    df_res = pd.DataFrame(all_results)
    avg_res = df_res.mean()
    std_res = df_res.std()

    print("-" * 50)
    print(
        f"{'Average':<8} | {avg_res['MAE']:<8.4f} | {avg_res['R2']:<8.4f} | {avg_res['RMSE']:<8.4f} | {avg_res['MAPE']:<8.4f} | {avg_res['GM']:<8.4f}")
    print(
        f"{'Std Dev':<8} | {std_res['MAE']:<8.4f} | {std_res['R2']:<8.4f} | {std_res['RMSE']:<8.4f} | {std_res['MAPE']:<8.4f} | {std_res['GM']:<8.4f}")
    print("=" * 50)

    # --- 生成总体的回归拟合图和置信区间图 ---
    print("\n[绘图] 正在生成所有 Seed 汇总的总体误差分布图...")
    plot_overall_figures(np.array(global_x_w), np.array(global_y_w), model_name="Overall_AllSeeds_revise")

    # --- 自动保存到 CSV 文件 ---
    save_csv_path = "test_results_summary_revise.csv"
    df_res.to_csv(save_csv_path, index=False)

    with open(save_csv_path, 'a') as f:
        f.write(
            f"\nAverage,{avg_res['MAE']:.4f},{avg_res['R2']:.4f},{avg_res['RMSE']:.4f},{avg_res['MAPE']:.4f},{avg_res['GM']:.4f}")

    print(f"\n✅ 所有数据已保存至: {os.path.abspath(save_csv_path)}")
    print(f"✅ 总体趋势图已保存至: results 文件夹下 (后缀含 Overall_AllSeeds_revise)")
    print(f"总耗时: {time.time() - start_time:.1f}s")