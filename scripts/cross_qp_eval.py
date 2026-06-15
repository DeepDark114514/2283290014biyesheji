import os

import sys
import json
import csv
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG
from models import PureResUNet, DegFiLMResUNet
from losses import L1SSIMLoss
from datasets import build_dataloader
from train import validate
from utils.metrics import calc_psnr


def parse_args():
    parser = argparse.ArgumentParser(description="跨QP泛化评估")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--model_type", type=str, default='A', choices=['A', 'B'],
                        help="模型类型: A=PureResUNet, B=DegFiLMResUNet")
    parser.add_argument("--qp_list", nargs='+', type=int, default=[22, 27, 32, 37, 42],
                        help="测试QP列表")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"],
                        help="评估数据集")
    parser.add_argument("--out_dir", type=str, default="result/cross_qp", help="输出目录")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    return parser.parse_args()


@torch.no_grad()
def eval_single_qp(model, dataloader, device, loss_fn):
    # 在单个QP上评估，返回模型PSNR/SSIM/Loss 和 输入baseline PSNR/SSIM
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_loss = 0.0
    total_baseline_psnr = 0.0
    total_baseline_ssim = 0.0
    total_frames = 0

    from train import _pad_to_multiple
    from datasets.inference_utils import tile_predict
    from tqdm import tqdm

    for lq, hq, _ in tqdm(dataloader, desc=f'Eval QP', leave=False):
        lq = lq.to(device)
        hq = hq.to(device)
        _, _, h, w = lq.shape

        use_tile = (h > 720 or w > 1280)

        if use_tile:
            pred = tile_predict(model, lq, tile_size=CONFIG['patch_size'], stride=CONFIG['patch_size'] // 2)
        else:
            lq_padded, pads = _pad_to_multiple(lq, multiple=16)
            pred = model(lq_padded)
            pht, phb, pwl, pwr = pads
            pred = pred[:, :, pht:pht + h, pwl:pwl + w]

        # 模型输出指标
        loss, _, _, ssim_val = loss_fn(pred.clamp(0, 1), hq, return_components=True)
        total_psnr += float(calc_psnr(pred.clamp(0, 1), hq))
        total_ssim += ssim_val
        total_loss += loss.item()

        # 输入 baseline 指标（LQ vs GT）
        total_baseline_psnr += float(calc_psnr(lq.clamp(0, 1), hq))
        _, _, _, baseline_ssim_val = loss_fn(lq.clamp(0, 1), hq, return_components=True)
        total_baseline_ssim += baseline_ssim_val

        total_frames += 1

    if total_frames == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        total_psnr / total_frames,
        total_ssim / total_frames,
        total_loss / total_frames,
        total_baseline_psnr / total_frames,
        total_baseline_ssim / total_frames,
    )


def plot_degradation_curve(results, out_dir, model_label='A'):
    # 绘制QP退化曲线
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qp_list = [r['qp'] for r in results]
    psnr_list = [r['psnr'] for r in results]
    ssim_list = [r['ssim'] for r in results]
    base_psnr_list = [r['baseline_psnr'] for r in results]
    base_ssim_list = [r['baseline_ssim'] for r in results]
    delta_psnr_list = [r['delta_psnr'] for r in results]
    delta_ssim_list = [r['delta_ssim'] for r in results]

    # PSNR曲线
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(qp_list, psnr_list, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax.set_xlabel('QP（压缩强度）', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title(f'{model_label}方案：跨QP泛化性能退化曲线', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(qp_list)
    for x, y in zip(qp_list, psnr_list):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / 'psnr_degradation_curve.png', dpi=300)
    plt.close(fig)

    # SSIM曲线
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(qp_list, ssim_list, marker='s', linewidth=2, markersize=8, color='#A23B72')
    ax.set_xlabel('QP（压缩强度）', fontsize=12)
    ax.set_ylabel('SSIM', fontsize=12)
    ax.set_title(f'{model_label}方案：跨QP泛化性能退化曲线', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(qp_list)
    for x, y in zip(qp_list, ssim_list):
        ax.annotate(f'{y:.4f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / 'ssim_degradation_curve.png', dpi=300)
    plt.close(fig)

    # Gain对比柱状图：Model vs Baseline PSNR
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(qp_list))
    width = 0.35
    bars1 = ax.bar(x - width/2, base_psnr_list, width, label='Baseline（输入LQ）', color='#A23B72', alpha=0.8)
    bars2 = ax.bar(x + width/2, psnr_list, width, label='模型输出', color='#2E86AB', alpha=0.8)
    ax.set_xlabel('QP（压缩强度）', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title(f'{model_label}方案：模型输出 vs 输入基线 PSNR 对比', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(qp_list)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    # 标注两个柱子的具体数值
    for i, (bp, mp) in enumerate(zip(base_psnr_list, psnr_list)):
        ax.annotate(f'{bp:.2f}', (x[i] - width/2, bp), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=8, color='#A23B72')
        ax.annotate(f'{mp:.2f}', (x[i] + width/2, mp), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=8, color='#2E86AB')
    fig.tight_layout()
    fig.savefig(out_dir / 'gain_comparison.png', dpi=300)
    plt.close(fig)

    # Gain波动曲线：突出增益随QP的变化
    fig, ax1 = plt.subplots(figsize=(9, 5))
    color1 = '#2E86AB'
    ax1.set_xlabel('QP（压缩强度）', fontsize=12)
    ax1.set_ylabel('PSNR 增益 (dB)', color=color1, fontsize=12)
    line1 = ax1.plot(qp_list, delta_psnr_list, marker='o', linewidth=2.5, markersize=10, color=color1, label='PSNR 增益')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(qp_list)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    # 填充区域突出波动
    ax1.fill_between(qp_list, delta_psnr_list, alpha=0.2, color=color1)
    for x, y in zip(qp_list, delta_psnr_list):
        ax1.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 12), ha='center', fontsize=10, color=color1, fontweight='bold')

    ax2 = ax1.twinx()
    color2 = '#A23B72'
    ax2.set_ylabel('SSIM 增益', color=color2, fontsize=12)
    line2 = ax2.plot(qp_list, delta_ssim_list, marker='s', linewidth=2.5, markersize=10, color=color2, label='SSIM 增益')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.fill_between(qp_list, delta_ssim_list, alpha=0.2, color=color2)
    for x, y in zip(qp_list, delta_ssim_list):
        ax2.annotate(f'{y:.4f}', (x, y), textcoords="offset points", xytext=(0, -18), ha='center', fontsize=10, color=color2, fontweight='bold')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    ax1.set_title(f'{model_label}方案：跨QP增益波动（模型输出 - 输入基线）', fontsize=14)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'gain_fluctuation_curve.png', dpi=300)
    plt.close(fig)

    # 合并图
    fig, ax1 = plt.subplots(figsize=(9, 5))
    color1 = '#2E86AB'
    ax1.set_xlabel('QP（压缩强度）', fontsize=12)
    ax1.set_ylabel('PSNR (dB)', color=color1, fontsize=12)
    line1 = ax1.plot(qp_list, psnr_list, marker='o', linewidth=2, markersize=8, color=color1, label='PSNR')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(qp_list)

    ax2 = ax1.twinx()
    color2 = '#A23B72'
    ax2.set_ylabel('SSIM', color=color2, fontsize=12)
    line2 = ax2.plot(qp_list, ssim_list, marker='s', linewidth=2, markersize=8, color=color2, label='SSIM')
    ax2.tick_params(axis='y', labelcolor=color2)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    ax1.set_title(f'{model_label}方案：跨QP性能退化曲线', fontsize=14)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'combined_degradation_curve.png', dpi=300)
    plt.close(fig)

    print(f"[OK] 图表已保存至 {out_dir}")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print(f"[INFO] 加载模型: {args.model_path}")
    if args.model_type == 'B':
        model = DegFiLMResUNet(base_ch=CONFIG['base_ch'])
    else:
        model = PureResUNet(base_ch=CONFIG['base_ch'])
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    loss_fn = L1SSIMLoss(l1_weight=CONFIG['l1_weight'], ssim_weight=CONFIG['ssim_weight']).to(device)

    # 断点续传：加载已有结果
    json_path = out_dir / 'cross_qp_results.json'
    csv_path = out_dir / 'cross_qp_results.csv'
    results = []
    completed_qps = set()
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        completed_qps = {r['qp'] for r in results}
        print(f"[INFO] 已加载已有结果，已完成QP: {sorted(completed_qps)}")

    # 逐个QP评估
    print(f"[INFO] 开始跨QP评估，QP列表: {args.qp_list}")
    for qp in args.qp_list:
        if qp in completed_qps:
            print(f"\n[Skip] QP{qp} 已评估，跳过")
            continue
        print(f"\n[Eval] QP{qp} ...")
        dataloader = build_dataloader(CONFIG, args.split, qp=qp)
        psnr, ssim, loss, base_psnr, base_ssim = eval_single_qp(model, dataloader, device, loss_fn)
        results.append({
            'qp': qp,
            'psnr': psnr, 'ssim': ssim, 'loss': loss,
            'baseline_psnr': base_psnr, 'baseline_ssim': base_ssim,
            'delta_psnr': psnr - base_psnr, 'delta_ssim': ssim - base_ssim,
        })
        print(f"  QP{qp}: Model PSNR={psnr:.4f} dB, SSIM={ssim:.4f}")
        print(f"         Baseline PSNR={base_psnr:.4f} dB, SSIM={base_ssim:.4f}")
        print(f"         Gain PSNR={psnr-base_psnr:+.4f} dB, SSIM={ssim-base_ssim:+.4f}")
        
        # 立即保存（断点续传）
        results_sorted = sorted(results, key=lambda x: x['qp'])
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_sorted, f, indent=2, ensure_ascii=False)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['qp', 'psnr', 'ssim', 'loss', 'baseline_psnr', 'baseline_ssim', 'delta_psnr', 'delta_ssim'])
            writer.writeheader()
            writer.writerows(results_sorted)
        print(f"  [OK] 中间结果已保存")

    # 画退化曲线
    plot_degradation_curve(results, out_dir, model_label=args.model_type)

    # 打印退化分析
    print("\n" + "=" * 60)
    print("跨QP退化分析摘要")
    print("=" * 60)
    base_qp = 32
    base_psnr = next(r['psnr'] for r in results if r['qp'] == base_qp)
    base_ssim = next(r['ssim'] for r in results if r['qp'] == base_qp)
    print(f"{'QP':>4} {'Model PSNR':>12} {'Base PSNR':>12} {'Gain':>10} {'Model SSIM':>12} {'Base SSIM':>12} {'Gain':>10}")
    print("-" * 78)
    for r in sorted(results, key=lambda x: x['qp']):
        delta_psnr = r['psnr'] - base_psnr
        delta_ssim = r['ssim'] - base_ssim
        print(f"{r['qp']:4d} {r['psnr']:12.4f} {r['baseline_psnr']:12.4f} {r['delta_psnr']:+10.4f} "
              f"{r['ssim']:12.4f} {r['baseline_ssim']:12.4f} {r['delta_ssim']:+10.4f}")
    print("=" * 78)
    print(f"注: Gain = Model - Baseline (输入 LQ 本身的指标)")


if __name__ == '__main__':
    main()
