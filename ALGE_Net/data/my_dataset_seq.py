# my_dataset_seq.py
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pydicom as dicom
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils1 import read_csv_random_split_data

# ================= 论文绘图风格设置 =================
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.facecolor'] = '#F1F4F6'  # 浅灰蓝色背景
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = 'white'  # 白色网格线
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.linewidth'] = 1.2
plt.rcParams['axes.spines.top'] = False  # 移除边框
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams['axes.axisbelow'] = True  # 确保网格线在图表元素下方



class MyDataSetSeq(Dataset):
    """
    预测时按切片加载的数据集（每个样本 = 一个切片）
    seq_df 来自 group_by_sequence 的输出。
    """

    def __init__(self, images_path: list, data_dir, batch_size, seq_df, transform=None):
        self.samples = []
        for _, row in seq_df.iterrows():
            seq_id = row['seq_id']
            label = row['day']  # 胎龄标签（统一字段名）
            for img_path, left, right, top, bottom in zip(
                    row['images'], row['lefts'], row['rights'], row['tops'], row['bottoms']
            ):
                self.samples.append({
                    "img_path": img_path,
                    "seq_id": seq_id,
                    "label": label,
                    "left": left,
                    "right": right,
                    "top": top,
                    "bottom": bottom
                })
        self.transform = transform

    def __len__(self):
        # 返回所有切片数量，而非序列数
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        img_path = sample["img_path"]
        seq_id = sample["seq_id"]
        label = np.asarray([sample["label"]], dtype=np.float32)

        # --- 读取 DICOM ---
        dicom_file = dicom.dcmread(img_path)
        image = dicom_file.pixel_array
        image_normalized = (image - image.min()) / (image.max() - image.min() + 1e-8)
        image_8bit = (image_normalized * 255).astype(np.uint8)

        # --- ROI 裁剪 ---
        h, w = image_8bit.shape[:2]
        left_i, right_i = max(0, int(sample["left"])), min(w, int(sample["right"]))
        top_i, bottom_i = max(0, int(sample["top"])), min(h, int(sample["bottom"]))
        roi = image_8bit[top_i:bottom_i, left_i:right_i] if bottom_i > top_i and right_i > left_i else image_8bit

        img = Image.fromarray(roi).convert('L')

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        # 每个样本返回一张切片图像，而不是整组
        return img, label, seq_id

    class collater():
        def __init__(self, *params):
            self.params = params

        def __call__(self, batch):
            images, labels, seqs = tuple(zip(*batch))
            labels = torch.as_tensor(np.array(labels))
            return list(images), labels, seqs

    def preprocess(self, images):
        image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        max_h = max([s[0] for s in image_sizes])
        max_w = max([s[1] for s in image_sizes])
        batch_shape = (len(images), images[0].shape[0], max_h, max_w)

        padded_images = images[0].new_full(batch_shape, 0.0)
        for i, img in enumerate(images):
            c, h, w = img.shape
            padded_images[i, :, :h, :w] = img

        return padded_images


def visualize_dataset_features(full_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Figure 2，三张图导出为 PDF：
      figure_2a.pdf - Participant screening flow（简洁 cohort flowchart）
      figure_2b.pdf - Dataset hierarchy（participant / sequence / slice 层级图）
      figure_2c.pdf - Train/Test FGA distribution（participant-level 箱线图，按 fs 去重）
    """
    import os
    import matplotlib.pyplot as plt 

    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    theme_color = '#A5C1D9'

    # --- 图1: Figure 2a Participant screening（三柱，excluded 放最右侧） ---
    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    labels_a = ['screened', 'included', 'excluded']
    vals_a = [272, 207, 65]
    x_a = np.arange(len(labels_a))
    # 三根柱统一颜色
    ax.bar(x_a, vals_a, width=0.55, color=theme_color, edgecolor='none')
    # 柱顶数字
    for xi, v in zip(x_a, vals_a):
        ax.text(xi, v + 6, f'{v:,}', ha='center', va='bottom', fontsize=13, color='#2D3B45')
    # excluded 原因直接黑字写在 excluded 柱上方（相对柱顶再往上移 30，避免盖住 65 数字）
    ax.text(2, vals_a[2] + 30,
            '1 fetal malposition\n2 fetal anomaly\n62 developmental anomalies',
            ha='center', va='bottom', fontsize=11, color='#2D3B45',
            linespacing=1.3)
    ax.set_xticks(x_a)
    ax.set_xticklabels(labels_a, fontsize=12)
    ax.set_ylabel('Participants', fontsize=14)
    ax.set_xlim(-0.6, 2.6)
    ax.tick_params(axis='both', which='both', length=0, labelsize=12)
    ax.set_ylim(0, 300)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'figure_2a.pdf'), format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()

    # --- 图2: Figure 2b Dataset hierarchy（柱状图，风格同 2c） ---
    # 数字自动从 full_df 计算（participant / sequence / slice 三层）
    n_participants = int(full_df['fs'].nunique())
    # 序列数按 image 路径计算：先按 (fs, series 文件夹) 分组，组内每 5 张连续切片折 1 个 sequence。
    # 这样 2 个含 10 张图的系列（各为两组 5 张连续）各折成 2 个 sequence，共 326，
    
    valid_df = full_df.dropna(subset=['image', 'fs']).copy()
    valid_df['_seq'] = valid_df['image'].apply(
        lambda x: os.path.basename(os.path.dirname(x)) if isinstance(x, str) else '')
    seq_counts = valid_df.groupby(['fs', '_seq']).size()
    slices_per_seq = 5  # 固定设计常量：每序列选取 5 张连续冠状面
    n_sequences = int((seq_counts // slices_per_seq).sum())
    n_slices = n_sequences * slices_per_seq

    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    labels_b = ['participants', 'MRI sequences', 'slices']
    vals_b = [n_participants, n_sequences, n_slices]
    x_b = np.arange(len(labels_b))
    ax.bar(x_b, vals_b, width=0.55, color=theme_color, edgecolor='none')
   
    for xi, v in zip(x_b, vals_b):
        ax.text(xi, v + 30, f'{v:,}', ha='center', va='bottom', fontsize=13, color='#2D3B45')

    ax.text(1.0, n_sequences + 150,
            '5 consecutive coronal\nslices / sequence',
            ha='center', va='bottom', fontsize=11, color='#2D3B45',
            linespacing=1.2)
    ax.set_xticks(x_b)
    ax.set_xticklabels(labels_b, fontsize=12)
    ax.set_ylabel('Count', fontsize=14)
    ax.tick_params(axis='both', which='both', length=0, labelsize=12)
    ax.set_ylim(0, 1800)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'figure_2b.pdf'), format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()

    # --- 图3: Figure 2c 统计学对比 箱线图 (Training vs Test, participant-level) ---
    # train/test split 已是 participant-level，绘图前按 participant ID (fs) 去重
    if 'fs' in train_df.columns and 'fs' in test_df.columns:
        train_part_df = train_df.drop_duplicates('fs')
        test_part_df = test_df.drop_duplicates('fs')
    else:
        print("Warning: 'fs' column not found, falling back to row-level statistics.")
        train_part_df, test_part_df = train_df, test_df
    print(f"Training: {len(train_part_df)} participants, Testing: {len(test_part_df)} participants")
    train_part_weeks = train_part_df['day'] // 7
    test_part_weeks = test_part_df['day'] // 7

    plt.figure(figsize=(6, 5))
    bplot = plt.boxplot([train_part_weeks, test_part_weeks],
                        tick_labels=['Training Set', 'Test Set'],
                        patch_artist=True,
                        widths=0.3,
                        boxprops=dict(facecolor=theme_color, edgecolor='none'),
                        whiskerprops=dict(color='gray', linewidth=1.5),
                        capprops=dict(color='gray', linewidth=1.5),
                        medianprops=dict(color='#2D3B45', linewidth=2),
                        flierprops=dict(marker='o', markerfacecolor=theme_color, markeredgecolor='none', alpha=0.6))

    plt.ylabel('FGA (weeks)', fontsize=14)
    plt.tick_params(axis='both', which='both', length=0, labelsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'figure_2c.pdf'), format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()

    print(f"Figure 2 (2a/2b/2c) PDFs with enlarged fonts and tighter margins generated in: {results_dir}")


if __name__ == "__main__":
    import os

    DATA_CSV = r"./fetal_data/newdataWindowU1.csv"

    if not os.path.isabs(DATA_CSV):
        DATA_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_CSV)

    print(f"Loading real data from: {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)

    if 'week' not in df.columns:
        raise ValueError(
            f"'week' column not found in {DATA_CSV}. "
            "read_csv_random_split_data requires columns: fs, week, day, ..."
        )

    full_df = pd.concat([train_df, test_df], ignore_index=True)

    print(
        f"Real data loaded: {len(train_df.drop_duplicates('fs'))} train participants, "
        f"{len(test_df.drop_duplicates('fs'))} test participants"
    )

    train_df, test_df = read_csv_random_split_data(df)

    visualize_dataset_features(full_df=full_df, train_df=train_df, test_df=test_df)
