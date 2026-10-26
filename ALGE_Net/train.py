import os
import sys
import time
import random
import argparse
import warnings

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.my_dataset import MyDataSet
from utils.utils2 import AverageMeter
from utils.utils1 import read_csv_random_split_data, set_seed
from models.Dual_branch_model import DualBranchModel  # 双分支模型

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def compute_edge_target_from_sobel(images):
    """
    生成边缘伪标签
    """
    device = images.device
    B, C, H, W = images.shape

    # Sobel kernel
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device).view(1, 1, 3, 3)

    # 计算梯度
    gx = F.conv2d(images, sobel_x, padding=1)
    gy = F.conv2d(images, sobel_y, padding=1)
    grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    grad_flat = grad_mag.view(B, -1)

    # 计算每张图的 2% 和 98% 分位数作为 min 和 max
    # 注意：torch.quantile 需要 PyTorch 1.7+
    v_min = torch.quantile(grad_flat, 0.02, dim=1, keepdim=True)
    v_max = torch.quantile(grad_flat, 0.98, dim=1, keepdim=True)

    # 截断异常值 (Outliers)
    grad_mag = torch.clamp(grad_mag, min=v_min.view(B, 1, 1, 1), max=v_max.view(B, 1, 1, 1))

    # 归一化到 0~1
    edge_target = (grad_mag - v_min.view(B, 1, 1, 1)) / (v_max.view(B, 1, 1, 1) - v_min.view(B, 1, 1, 1) + 1e-8)

    return edge_target


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    df = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}.csv"))
    train_images_path, _ = read_csv_random_split_data(df)

    img_size = 96
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=16),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=(0, 360), translate=(0.2, 0.2)),
            transforms.Normalize([0.326930], [0.129822])
        ])
    }

    train_dataset = MyDataSet(
        data_dir=args.data_dir,
        images_path=train_images_path,
        transform=data_transform["train"],
        batch_size=args.batch_size
    )
    print(f"train dataset : {len(train_dataset)} images")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        collate_fn=MyDataSet.collater()
    )

    model = DualBranchModel(
        small_input=True,
        dropout=None,
        fusion_mode="all",
        lambda_edge=args.lambda_edge,
        lambda_cons=args.lambda_cons,
        lambda_sobel=args.lambda_sobel 
    )
    model = model.to(device)

    losses_total = AverageMeter('Loss_total', ':.3f')
    losses_age = AverageMeter('Loss_age', ':.3f')
    losses_edge = AverageMeter('Loss_edge', ':.3f')
    losses_cons = AverageMeter('Loss_cons', ':.3f')
    losses_sobel = AverageMeter('Loss_sobel', ':.3f')  

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=1e-5) if args.optimizer == 'adam' else optim.SGD(params,
                                                                                                             lr=args.lr,
                                                                                                             momentum=0.9,
                                                                                                             weight_decay=1e-4)

    Best_MSE = 1e5
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        losses_total.reset();
        losses_age.reset();
        losses_edge.reset();
        losses_cons.reset();
        losses_sobel.reset()

        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, (images, labels) in enumerate(train_bar):
            images = images.to(device)
            labels = labels.to(device).float()

            # 生成边缘标签供原本的 edge_loss 使用
            edge_target = compute_edge_target_from_sobel(images)

            # 前向传播：模型内部现在会自动计算 loss_sobel
            out = model(images, age_gt=labels, edge_target=edge_target)

            loss = out["loss_total"]
            loss_age = out["loss_age"]
            loss_edge = out["loss_edge"]
            loss_cons = out["loss_cons"]
            loss_sobel_val = out.get("loss_sobel", torch.tensor(0.0))  

            assert not (np.isnan(loss.item()) or loss.item() > 1e6), f"Loss explosion: {loss.item()}"

            batch_size = images.size(0)
            losses_total.update(loss.item(), batch_size)
            losses_age.update(loss_age.item(), batch_size)
            losses_edge.update(loss_edge.item(), batch_size)
            losses_cons.update(loss_cons.item(), batch_size)
            losses_sobel.update(loss_sobel_val.item(), batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = (
                f"epoch[{epoch + 1}/{args.epochs}] "
                f"total:{loss:.3f} age:{loss_age:.3f} "
                f"sob:{loss_sobel_val:.3f}"  
            )

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        if avg_loss < Best_MSE:
            Best_MSE = avg_loss
            torch.save(model.state_dict(), args.save_path)

        print(
            f"Epoch {epoch + 1}: avg_total={avg_loss:.3f}, age={losses_age.avg:.3f}, edge={losses_edge.avg:.3f}, sobel={losses_sobel.avg:.3f}")

    print(f"Final Best_Avg_Loss={Best_MSE:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data_dir', type=str, default='D://Yulia//KANResnet//HSResNet_base//fetal_data')
    parser.add_argument('--dataset', type=str, default='newdataWindowU1')
    parser.add_argument('--save_path', type=str, default='')

    parser.add_argument('--lambda_edge', type=float, default=0.5)
    parser.add_argument('--lambda_cons', type=float, default=0.1)
    parser.add_argument('--lambda_sobel', type=float, default=0.2, help='weight for sobel self-supervised loss')

    opt = parser.parse_args()

    # 服务器运行 10 seeds 循环
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        set_seed(seed)
        start_time = time.time()
        opt.save_path = f'D://Yulia//KANResNet//HSResNet_base//ED-KANet//weights_dual//model-best-{seed}-1.pth'
        print(f">>> Training Seed: {seed}")
        main(opt)
        print(f"Seed {seed} train_run_time: {time.time() - start_time:.2f}s\n" + "=" * 50)