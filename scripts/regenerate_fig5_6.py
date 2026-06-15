# 重新生成图5-6 多QP全局对比，PSNR精确到小数点后三位

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from config import CONFIG
from models import PureResUNet, DegFiLMResUNet
from datasets.yuv_io import read_yuv

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_yuv_name(name: str):
    parts = name.replace('.yuv', '').split('_')
    w, h = parts[-3].split('x') if 'x' in parts[-3] else parts[-2].split('x')
    frames = parts[-2] if 'x' in parts[-3] else parts[-1]
    if 'qp' in frames:
        frames = parts[-3]
    return int(w), int(h), int(frames)


@torch.no_grad()
def inference(model, x, device):
    model.eval()
    x = x.to(device)
    _, _, h, w = x.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h or pad_w:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    pred = model(x)
    pred = torch.clamp(pred, 0, 1)
    if pad_h or pad_w:
        pred = pred[:, :, :h, :w]
    pred = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return pred


def calc_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    out_dir = Path('docs/figures')
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print('[INFO] Loading A (PureResUNet)...')
    model_a = PureResUNet(base_ch=CONFIG['base_ch']).to(device)
    ckpt_a = torch.load('checkpoints/best_model.pth', map_location=device)
    state_a = ckpt_a.get('model_state_dict', ckpt_a)
    model_a.load_state_dict(state_a)

    print('[INFO] Loading B (DegFiLMResUNet)...')
    model_b = DegFiLMResUNet(base_ch=CONFIG['base_ch']).to(device)
    ckpt_b = torch.load('logs/B_20260519_121523/best_model.pth', map_location=device)
    state_b = ckpt_b.get('model_state_dict', ckpt_b)
    model_b.load_state_dict(state_b)

    seq_name = 'BasketballPass_416x240_500'
    frame_idx = 387
    qps = [22, 32, 42]

    gt_path = Path(CONFIG['dataset_root']) / 'gt' / 'test' / f'{seq_name}.yuv'
    w, h, _ = parse_yuv_name(gt_path.name)
    gt_seq = read_yuv(str(gt_path), w, h)
    gt = gt_seq[frame_idx]

    # 收集所有结果
    rows = []
    for qp in qps:
        lq_path = Path(CONFIG['dataset_root']) / 'compressed' / 'test' / f'{seq_name}_qp{qp}.yuv'
        lq_seq = read_yuv(str(lq_path), w, h)
        lq = lq_seq[frame_idx]
        lq_t = torch.from_numpy(lq).permute(2, 0, 1).unsqueeze(0).float()

        pred_a = inference(model_a, lq_t, device)
        pred_b = inference(model_b, lq_t, device)

        psnr_lq = calc_psnr(lq, gt)
        psnr_a = calc_psnr(pred_a, gt)
        psnr_b = calc_psnr(pred_b, gt)

        rows.append({
            'qp': qp,
            'lq': lq, 'gt': gt,
            'pred_a': pred_a, 'pred_b': pred_b,
            'psnr_lq': psnr_lq,
            'psnr_a': psnr_a,
            'psnr_b': psnr_b,
        })
        print(f'  QP{qp}: LQ={psnr_lq:.3f}dB, A={psnr_a:.3f}dB, B={psnr_b:.3f}dB')

    # 绘图：3行4列
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    fig.suptitle(f'{seq_name}    第 {frame_idx} 帧', fontsize=14, y=0.98, fontweight='bold')

    col_titles = ['压缩后的输入', '压缩前的原图', 'A方案修复图', 'B方案修复图']
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=12, pad=8, fontweight='bold')

    for row_idx, r in enumerate(rows):
        imgs = [r['lq'], r['gt'], r['pred_a'], r['pred_b']]
        psnr_text = f'QP={r["qp"]}'

        for col_idx, (ax, img) in enumerate(zip(axes[row_idx], imgs)):
            ax.imshow(np.clip(img, 0, 1))
            ax.axis('off')

            # 在第一列左侧添加QP和PSNR文字
            if col_idx == 0:
                ax.text(
                    -0.02, 0.5, psnr_text,
                    transform=ax.transAxes,
                    fontsize=12, verticalalignment='center',
                    horizontalalignment='right', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, edgecolor='none')
                )

    plt.tight_layout(rect=[0.04, 0.0, 1.0, 0.95])
    out_path = out_dir / '图5-6 多QP全局对比.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\n[OK] Saved to {out_path}')


if __name__ == '__main__':
    main()
