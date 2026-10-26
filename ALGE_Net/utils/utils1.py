import random
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import os  # 添加os模块导入
from scipy.stats import gmean
from pandas import Series
from sklearn.linear_model import LinearRegression
from sympy.physics.quantum.identitysearch import scipy
from torch import nn
from tqdm import tqdm
from utils.reg_R2 import *
from utils.utils2 import AverageMeter

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def set_seed(seed):
    """设置全面的随机种子，确保实验可复现"""
    random.seed(seed)  # Python内置的随机数生成器
    np.random.seed(seed)  # NumPy的随机数生成器
    torch.manual_seed(seed)  # PyTorch的CPU随机数生成器
    torch.cuda.manual_seed(seed)  # PyTorch的GPU随机数生成器
    torch.cuda.manual_seed_all(seed)  # 多GPU环境下的随机数生成器

    # 确保CuDNN算法的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 可能会降低性能，但保证结果一致

    # 设置环境变量，确保其他库的随机数生成也一致
    os.environ['PYTHONHASHSEED'] = str(seed)

def read_csv_random_split_data(df):
    train_images_path = pd.DataFrame()
    test_images_path = pd.DataFrame()


    # 1.获取csv文件week列的数据
    weeks = df[['week']]
    unique_weeks = weeks['week'].unique()

    for week in unique_weeks:
        week_df = df[df['week'] == week]

        # 2.根据week去查看有几个对应的病人
        week_patients = week_df['fs'].unique()

        # 3.根据week和病人数量划分训练集和测试集
        if week == 22:
            train, test = train_test_split(week_patients, train_size=1 / 2)
        elif week == 23:
            train, test = train_test_split(week_patients, train_size=8 / 10)
        elif week == 24:
            train, test = train_test_split(week_patients, train_size=12 / 15)
        elif week == 25:
            train, test = train_test_split(week_patients, train_size=7 / 9)
        elif week == 26:
            train, test = train_test_split(week_patients, train_size=4 / 5)
        elif week == 27:
            train, test = train_test_split(week_patients, train_size=8 / 10)
        elif week == 28:
            train, test = train_test_split(week_patients, train_size=11 / 14)
        elif week == 29:
            train, test = train_test_split(week_patients, train_size=18 / 22)
        elif week == 30:
            train, test = train_test_split(week_patients, train_size=8 / 10)
        elif week == 31:
            train, test = train_test_split(week_patients, train_size=11 / 14)
        elif week == 32:
            train, test = train_test_split(week_patients, train_size=15 / 19)
        elif week == 33:
            train, test = train_test_split(week_patients, train_size=13 / 16)
        elif week == 34:
            train, test = train_test_split(week_patients, train_size=20 / 24)
        elif week == 35:
            train, test = train_test_split(week_patients, train_size=10 / 13)
        elif week == 36:
            train, test = train_test_split(week_patients, train_size=10 / 13)
        elif week == 37:
            train, test = train_test_split(week_patients, train_size=7 / 9)
        elif week == 38:
            train, test = train_test_split(week_patients, train_size=1 / 2)
        else:
            continue  # 如果不是指定的孕周，跳过
      
        # 4.更新DataFrame
        week_df.loc[week_df['fs'].isin(train), 'split'] = '0'  # 训练集
        week_df.loc[week_df['fs'].isin(test), 'split'] = '1'  # 测试集

        # 5.合并到总的DataFrame
        train_images_path = pd.concat([train_images_path, week_df[week_df['split'] == '0']])
        test_images_path = pd.concat([test_images_path, week_df[week_df['split'] == '1']])

    return train_images_path, test_images_path


@torch.no_grad()
def evaluateR2_week_seaborn(model, data_loader, device, batch_size):
    model.eval()
    criterion_l1 = nn.L1Loss()

    losses_l1 = AverageMeter('Loss (L1)', ':.3f')
    metric_rmse = AverageMeter('rmse', ':.3f')
    metric_mape = AverageMeter('mape', ':.3f')


    data_loader = tqdm(data_loader, file=sys.stdout)
    # 真实标签
    x = []
    # 预测标签
    y = []

    GM_all = []


    for val_data in data_loader:
        val_images, val_labels = val_data

        outputs = model(val_images.to(device))
        val_labels = val_labels.to(device)

        # All
        loss_l1 = criterion_l1(outputs, val_labels)
        RMSE, MAPE, GM = metric_RMSE_MAPE_GM(outputs, val_labels)

        # 转列表
        outputs = outputs.tolist()
        val_labels = val_labels.tolist()

        if (len(val_labels) < batch_size):
            batch_size = len(val_labels)

        for i in range(batch_size):
            a = val_labels[i]
            b = outputs[i]
            a2 = (a[0] / 7) #将标签从天转化为周
            b2 = (b[0] / 7)

            x.append(a2)
            y.append(b2)

        # 更新计算metric
        # All
        losses_l1.update(loss_l1.item(), val_images.size(0))
        metric_rmse.update(RMSE.item(), val_images.size(0))
        metric_mape.update(MAPE.item(), val_images.size(0))
        GM_all.extend(GM.cpu().numpy())

    # =========================画测试误差分布图========================
    # 计算拟合度
    R2 = cal_R2(y, x)
    # 调整表格等距
    sns.set(style="darkgrid")
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = plt.gca()
    plt.gca().set_aspect(1)
    # 设置X轴标签
    plt.xlabel('Gestational age (week)', fontdict={'fontsize': 22})
    # 设置Y轴标签
    plt.ylabel('Predicted brain age (week)', fontdict={'fontsize': 22})
    # 设置坐标原点
    # 画表格线
    plt.grid(c='white')
    # 画直线
    list1 = []
    list2 = []
    # 22-36
    for i in range(0, 38):
        list1.append(i)
        list2.append(i)

    plt.plot(list1, list2, linewidth=3, color='black', linestyle='--', label='pred = truth')

    # ===========================回归=================================
    # 回归预测
    reg = LinearRegression()
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    # 拟合数据
    reg.fit(x, y)
    # 偏置：intercept_, 回归系数：coef_
    a, b = reg.intercept_, reg.coef_
    # 画回归拟合曲线
    # 对齐两端的点
    xx = np.vstack(([0], x))
    xx2 = np.vstack((xx, [37]))
    plt.plot(xx2, a + b * xx2, color="orangered", linewidth=3, label='output fit line')

    # ===========================散点==================================
    # 画散点
    plt.scatter(x, y, s=110, c='tomato', marker='.', label='test data')
    # # 设置横轴的上下限值
    plt.xlim(21.2, 36.5)
    # 设置纵轴的上下限值
    plt.ylim(21.2, 36.5)

    # 设置字体大小
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)

    # 画出空白的R2变量
    plt.plot(0, 0, color="#EAEAF2", label='R²=' + str(format(R2, '.3f')))

    # MAE
    MAE_week = format(losses_l1.avg / 7, '.3f')
    # MAE_week = format(losses_l1.avg, '.3f')

    plt.plot(0, 0, color="#EAEAF2", label='MAE=' + str(MAE_week))


    # RMSE, MAPE, GM
    RMSE_week = format(math.sqrt(metric_rmse.avg), '.3f')
    MAPE_week = format(metric_mape.avg, '.3f')

    GM_week = gmean(np.hstack(GM_all), axis=None).astype(float)
    GM_week = format(GM_week / 7, '.3f')

    # 显示线和点的信息到方框中
    plt.legend(fontsize=12.5)
    # plt.show()
    # Bland-Altman
    diff = y - x
    mean = np.mean([x, y], axis=0)
    # BlandAltman(diff, mean)

    return R2, MAE_week, RMSE_week, MAPE_week, GM_week

def metric_RMSE_MAPE_GM(outputs, val_labels):
    # 返回每个样本对应的误差
    criterion = nn.L1Loss(reduction='none')
    gm = criterion(outputs, val_labels)

    # RMSE
    errors = (outputs - val_labels) / 7
    squared_errors = errors ** 2

    absolute_errors = torch.abs((outputs - val_labels) / val_labels)

    rmse = torch.mean(squared_errors)

    # MAPE
    mape = torch.mean(absolute_errors) * 100

    return rmse, mape, gm


def group_by_sequence(df, data_dir=None, image_col='image'):
    """
    根据 CSV 的 image 路径自动提取序列（sequence）并分组。
    返回 seq_df：每行代表一个序列，包含：
        - fs: 患者 id
        - seq: 序列 id（取 image 的父目录名）
        - seq_path: 序列目录路径（可绝对或相对）
        - images: 该序列所有切片的完整路径列表
        - lefts, rights, tops, bottoms: ROI 坐标列表 (int)
        - week, day: 标签 (week/day)
    参数:
        df: pandas.DataFrame，包含至少列
            ['fs','image','XMIN','YMIN','XMAX','YMAX','week','day']
        data_dir: 如果 image 是相对路径，你可以传 data_dir 来拼接完整路径；
                  如果 image 已为绝对路径，请传 None（默认）。
    """
    df = df.copy()

    # 检查必要列
    required = [
        'fs', image_col, 'XMIN', 'YMIN',
        'XMAX', 'YMAX', 'week', 'day'
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"CSV 缺少必要列: {c}")

    # 规范化路径（如果需要）
    def make_path(p):
        if pd.isna(p):
            return None
        p = str(p)
        if not os.path.isabs(p) and data_dir:
            return os.path.join(data_dir, p)
        return p

    df[image_col] = df[image_col].apply(make_path)

    # 提取序列 ID（默认父文件夹名）
    df['seq'] = df[image_col].apply(
        lambda x: os.path.basename(os.path.dirname(x))
        if isinstance(x, str) else ''
    )
    df['seq_path'] = df[image_col].apply(
        lambda x: os.path.dirname(x) if isinstance(x, str) else ''
    )

    seq_df_list = []

    # 按 fs、seq、seq_path、week、day 分组
    grouped = df.groupby(
        ['fs', 'seq', 'seq_path', 'week', 'day'],
        dropna=False
    )

    for (fs, seq, seq_path, week, day), g in grouped:
        images = [
            p for p in g[image_col].tolist()
            if isinstance(p, str) and p != ''
        ]
        if len(images) == 0:
            continue

        lefts = [int(x) for x in g['XMIN'].tolist()]
        bottoms = [int(x) for x in g['YMIN'].tolist()]
        rights = [int(x) for x in g['XMAX'].tolist()]
        tops = [int(x) for x in g['YMAX'].tolist()]

        seq_df_list.append({
            'fs': fs,
            'seq': seq,
            'seq_path': seq_path,
            'seq_id': f"{fs}::{seq_path}",
            'images': images,
            'lefts': lefts,
            'rights': rights,
            'tops': tops,
            'bottoms': bottoms,
            'week': int(week) if pd.notna(week) else None,
            'day': float(day) if pd.notna(day) else None
        })

    seq_df = pd.DataFrame(seq_df_list)
    print(f"✅ 共解析出 {len(seq_df)} 个序列（每个序列包含若干切片）")
    return seq_df

@torch.no_grad()
def evaluateR2_week_seaborn_sequence(
        model,
        data_loader,
        device,
        batch_size,
        model_name="SSFN-Net",
        seed=1,
        save_format='pdf'):
    """
    基于序列（seq）的评估。

    1. 对每张切片进行模型预测。
    2. 同一序列内所有切片预测直接取算术平均。
    3. 不使用真实标签筛选、删除或加权切片。
    4. 在天单位下计算误差，最终以周为单位输出。

    说明：
    如果传入的 model 是 DualModelWrapper，
    则每张切片的原图和水平翻转图预测已在 Wrapper 内取平均。
    """

    model.eval()
    criterion_l1 = nn.L1Loss()
    data_loader = tqdm(data_loader, file=sys.stdout)

    results_per_slice = []

    # -------------------------------------------------
    # 1. 推理阶段：获取每张切片的预测值
    # -------------------------------------------------
    for val_images, val_labels, seq_ids in data_loader:
        for imgs_tensor, label, sid in zip(
                val_images, val_labels, seq_ids):

            imgs_tensor = imgs_tensor.to(device)
            slice_outputs = []

            # 对序列内每张切片分别进行前向传播
            for i in range(imgs_tensor.shape[0]):
                img = imgs_tensor[i]

                if img.dim() == 2:
                    img = img.unsqueeze(0).unsqueeze(0)
                elif img.dim() == 3:
                    img = img.unsqueeze(0)

                out = model(img)
                slice_outputs.append(out.item())

            # 记录逐切片预测和对应真实标签
            # 真实标签只用于最终评估，不参与预测聚合决策
            true_day = float(label.item())

            for slice_pred in slice_outputs:
                results_per_slice.append({
                    "seq": sid,
                    "true": true_day,
                    "pred": float(slice_pred)
                })

    # -------------------------------------------------
    # 2. 按序列聚合：GT-free arithmetic mean
    # -------------------------------------------------
  
    df_results = pd.DataFrame(results_per_slice)
    grouped_results = []

    if df_results.empty:
        raise RuntimeError(
            "No slice-level predictions were produced."
        )

    for seq, group in df_results.groupby("seq", sort=False):

        # 获取该序列所有切片的预测值
        slice_preds = group["pred"].to_numpy(
            dtype=np.float64
        )

        if len(slice_preds) == 0:
            continue

        # 所有切片预测直接取平均
        pred_mean = float(np.mean(slice_preds))

        # 真实标签只用于最终评估
        true_values = group["true"].to_numpy(
            dtype=np.float64
        )

        # 同一序列的切片应具有一致的真实标签
        if np.max(true_values) - np.min(true_values) > 1e-6:
            raise ValueError(
                f"Sequence {seq} has inconsistent "
                f"ground-truth labels: {true_values}"
            )

        true_mean = float(np.mean(true_values))

        # 保留原有字段，兼容后续指标和绘图代码
        grouped_results.append({
            "seq": seq,
            "true_mean": true_mean,
            "pred_mean": pred_mean,
            "n_used_slices": len(slice_preds),
            "n_total_slices": len(slice_preds)
        })

    df_seq = pd.DataFrame(grouped_results)

    if df_seq.empty:
        raise RuntimeError(
            "No sequence-level predictions were produced."
        )

    # -------------------------------------------------
    # 3. 计算整体评估指标
    # df_seq 中的 true_mean 和 pred_mean 均为天
    # -------------------------------------------------
    x_day = df_seq["true_mean"].to_numpy()
    y_day = df_seq["pred_mean"].to_numpy()

    # 转换为周
    x_w = x_day / 7.0
    y_w = y_day / 7.0

    # 评估指标（周）
    R2 = cal_R2(y_w.tolist(), x_w.tolist())
    MAE_week = np.mean(np.abs(y_w - x_w))
    RMSE_week = np.sqrt(
        np.mean((y_w - x_w) ** 2)
    )

    # MAPE
    eps = 1e-8
    MAPE_week = np.mean(
        np.abs((y_w - x_w) / (x_w + eps))
    ) * 100

    # GMAE
    abs_errors = np.abs(y_w - x_w)
    GM_week = np.exp(
        np.mean(np.log(abs_errors + eps))
    )

    # -------------------------------------------------
    # 4. 绘制测试集回归图
    # -------------------------------------------------
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)

    x = x_w
    y = y_w

    R2 = cal_R2(y.tolist(), x.tolist())

    sns.set_theme(
        style="darkgrid",
        rc={"axes.facecolor": "#eef2f6"}
    )

    fig = plt.figure(figsize=(6.5, 6.5))
    ax = plt.gca()
    plt.gca().set_aspect(1)

    plt.xlabel(
        'FGA (weeks)',
        fontdict={'fontsize': 18}
    )
    plt.ylabel(
        'Predicted FGA (weeks)',
        fontdict={'fontsize': 18}
    )

    plt.plot(
        [21, 39],
        [21, 39],
        linewidth=2.5,
        color='black',
        linestyle='--',
        label='pred = truth'
    )

    reg = LinearRegression()
    x_np = x.reshape(-1, 1)
    y_np = y.reshape(-1, 1)
    reg.fit(x_np, y_np)

    a, b = reg.intercept_, reg.coef_

    xx = np.array([21, 39]).reshape(-1, 1)

    plt.plot(
        xx,
        a + b * xx,
        color="#81b2d4",
        linewidth=2.5,
        label='output fit line'
    )

    plt.scatter(
        x,
        y,
        s=40,
        c='#a1c4d6',
        marker='o',
        alpha=0.9,
        label='test data'
    )

    plt.xlim(21, 39)
    plt.ylim(21, 39)
    plt.xticks(np.arange(22, 39, 2))
    plt.yticks(np.arange(22, 39, 2))

    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)

    plt.plot(
        [],
        [],
        ' ',
        label=f'$R^2 = {R2:.3f}$'
    )
    plt.plot(
        [],
        [],
        ' ',
        label=f'MAE = {MAE_week:.3f}'
    )

    plt.legend(
        loc='upper left',
        fontsize=13,
        frameon=True,
        facecolor='#f4f6f9',
        edgecolor='lightgray'
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            save_dir,
            f"figure7_regression_seq_{model_name}_seed{seed}.pdf"
        ),
        dpi=300
    )
    plt.close()

    # -------------------------------------------------
    # 5. 绘制 Bland-Altman 图
    # -------------------------------------------------
    diff = y_w - x_w
    mean_vals = (y_w + x_w) / 2.0

    BlandAltman(
        diff,
        mean_vals,
        save_path=os.path.join(
            save_dir,
            f"figure8_blandaltman_seq_{model_name}_seed{seed}.pdf"
        )
    )

    return (
        R2,
        MAE_week,
        RMSE_week,
        MAPE_week,
        GM_week,
        x_w,
        y_w
    )

def BlandAltman(
        diff,
        mean,
        save_path="results/figure8_blandaltman_seq.pdf"):

    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    upper_limit = mean_diff + 1.96 * std_diff
    lower_limit = mean_diff - 1.96 * std_diff

    sns.set_theme(
        style="darkgrid",
        rc={"axes.facecolor": "#eef2f6"}
    )

    fig = plt.figure(figsize=(9, 6))
    ax = plt.gca()

    plt.scatter(
        mean,
        diff,
        color='#a1c4d6',
        alpha=0.9,
        s=35
    )

    n = len(diff)
    confidence_level = 0.95
    degrees_of_freedom = n - 1

    t_value = abs(
        np.round(
            scipy.stats.t.ppf(
                (1 - confidence_level) / 2,
                degrees_of_freedom
            ),
            3
        )
    )

    s = np.std(diff)
    ci = t_value * s / np.sqrt(n)

    plt.axhspan(
        mean_diff - ci,
        mean_diff + ci,
        facecolor='#81b2d4',
        alpha=0.3
    )

    s1 = (
        1 / n
        + (1.96 ** 2) / (2 * (n - 1))
    ) * (s ** 2)

    ci1 = t_value * np.sqrt(s1)

    plt.axhspan(
        upper_limit - ci1,
        upper_limit + ci1,
        facecolor='#5a98c5',
        alpha=0.2
    )

    plt.axhspan(
        lower_limit - ci1,
        lower_limit + ci1,
        facecolor='#5a98c5',
        alpha=0.2
    )

    plt.axhline(
        0,
        color='black',
        linestyle='--',
        label='pred = truth',
        linewidth=2
    )

    mean_color = '#81b2d4'
    sd_color = '#5a98c5'

    plt.axhline(
        mean_diff,
        color=mean_color,
        label='Mean Difference',
        linewidth=2.5
    )

    plt.axhline(
        upper_limit,
        color=sd_color,
        linestyle='--',
        label='95% Limits of Agreement',
        linewidth=2.5
    )

    plt.axhline(
        lower_limit,
        color=sd_color,
        linestyle='--',
        linewidth=2.5
    )

    plt.xlabel(
        'Mean of Prediction and FGA (weeks)',
        fontdict={'fontsize': 18}
    )
    plt.ylabel(
        'Prediction - FGA (weeks)',
        fontdict={'fontsize': 18}
    )

    plt.xlim(21, 39)
    plt.ylim(-6.5, 6.5)
    plt.xticks(np.arange(22, 39, 2))
    plt.yticks(np.arange(-6, 7, 2))

    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)

    plt.legend(
        loc='upper left',
        fontsize=13,
        frameon=True,
        facecolor='#f4f6f9',
        edgecolor='lightgray'
    )

    text_x = plt.xlim()[1] - 0.2

    plt.text(
        text_x,
        upper_limit + 0.15,
        '+1.96 SD',
        ha='right',
        va='bottom',
        color='#2b4b66',
        fontsize=14
    )

    plt.text(
        text_x,
        upper_limit - 0.15,
        f'{upper_limit:.2f}',
        ha='right',
        va='top',
        color='#2b4b66',
        fontsize=14
    )

    plt.text(
        text_x,
        mean_diff + 0.15,
        'Mean',
        ha='right',
        va='bottom',
        color='#8b5a5a',
        fontsize=14
    )

    plt.text(
        text_x,
        mean_diff - 0.15,
        f'{mean_diff:.2f}',
        ha='right',
        va='top',
        color='#8b5a5a',
        fontsize=14
    )

    plt.text(
        text_x,
        lower_limit + 0.15,
        '-1.96 SD',
        ha='right',
        va='bottom',
        color='#2b4b66',
        fontsize=14
    )

    plt.text(
        text_x,
        lower_limit - 0.15,
        f'{lower_limit:.2f}',
        ha='right',
        va='top',
        color='#2b4b66',
        fontsize=14
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_overall_figures(
        x_w,
        y_w,
        save_dir='results',
        model_name="Overall_AllSeeds"):
    """
    接收所有 seed 汇总的真实值（x_w）和预测值（y_w），
    绘制总体回归拟合图和 Bland-Altman 图。
    """
    os.makedirs(save_dir, exist_ok=True)

    R2 = cal_R2(y_w.tolist(), x_w.tolist())
    MAE_week = np.mean(np.abs(y_w - x_w))

    # 绘制总回归拟合图
    sns.set_theme(
        style="darkgrid",
        rc={"axes.facecolor": "#eef2f6"}
    )

    fig = plt.figure(figsize=(6.5, 6.5))
    ax = plt.gca()
    plt.gca().set_aspect(1)

    plt.xlabel(
        'FGA (weeks)',
        fontdict={'fontsize': 18}
    )
    plt.ylabel(
        'Predicted FGA (weeks)',
        fontdict={'fontsize': 18}
    )

    plt.plot(
        [21, 39],
        [21, 39],
        linewidth=2.5,
        color='black',
        linestyle='--',
        label='pred = truth'
    )

    reg = LinearRegression()
    x_np = x_w.reshape(-1, 1)
    y_np = y_w.reshape(-1, 1)
    reg.fit(x_np, y_np)

    a, b = reg.intercept_, reg.coef_

    xx = np.array([21, 39]).reshape(-1, 1)

    plt.plot(
        xx,
        a + b * xx,
        color="#81b2d4",
        linewidth=2.5,
        label='output fit line'
    )

    plt.scatter(
        x_w,
        y_w,
        s=40,
        c='#a1c4d6',
        marker='o',
        alpha=0.9,
        label='test data'
    )

    plt.xlim(21, 39)
    plt.ylim(21, 39)
    plt.xticks(np.arange(22, 39, 2))
    plt.yticks(np.arange(22, 39, 2))

    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)

    plt.plot(
        [],
        [],
        ' ',
        label=f'$R^2 = {R2:.3f}$'
    )
    plt.plot(
        [],
        [],
        ' ',
        label=f'MAE = {MAE_week:.3f}'
    )

    plt.legend(
        loc='upper left',
        fontsize=13,
        frameon=True,
        facecolor='#f4f6f9',
        edgecolor='lightgray'
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            save_dir,
            f"figure7_regression_seq_{model_name}.pdf"
        ),
        dpi=300
    )
    plt.close()

    # 绘制总体 Bland-Altman 图
    diff = y_w - x_w
    mean_vals = (y_w + x_w) / 2.0

    BlandAltman(
        diff,
        mean_vals,
        save_path=os.path.join(
            save_dir,
            f"figure8_blandaltman_seq_{model_name}.pdf"
        )
    )


