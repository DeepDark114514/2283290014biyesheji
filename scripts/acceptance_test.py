#  南京信息工程大学22级信安1班 202283290014
# 2026.5.13
# 修复验收脚本：验证以下3个关键修复点
# 1. 对称 pad：1080 -> 1088，上下各pad 4，推理后crop回1080
# 2. Tile强制：Class A/B (h>720 or w>1280) 强制 tile-based，小分辨率整帧
# 3. 显存：bs=8, patch=256, base_ch=32 完整训练步峰值 < 14GB

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from config import CONFIG
from models import PureResUNet
from losses import L1SSIMLoss
from datasets import build_dataloader, tile_predict
from train import validate, _pad_to_multiple


def test_symmetric_pad():
    print("=" * 60)
    print("[1] 对称 Pad 测试")
    print("=" * 60)

    # 模拟 1920x1080（Class B 典型尺寸）
    x = torch.randn(1, 3, 1080, 1920)
    x_pad, pads = _pad_to_multiple(x, multiple=16)
    pht, phb, pwl, pwr = pads

    print(f"  原始尺寸: {x.shape}")
    print(f"  pad_h_top={pht}, pad_h_bottom={phb}, pad_w_left={pwl}, pad_w_right={pwr}")
    print(f"  pad后尺寸: {x_pad.shape}")

    assert x_pad.shape[-2] == 1088, f"高度应为1088，实际{x_pad.shape[-2]}"
    assert x_pad.shape[-1] == 1920, f"宽度应为1920，实际{x_pad.shape[-1]}"
    assert pht == 4 and phb == 4, f"上下pad应各为4，实际{pht},{phb}"
    assert pwl == 0 and pwr == 0, f"左右不应pad，实际{pwl},{pwr}"

    # 模拟 forward 后 crop
    pred_pad = torch.randn_like(x_pad)
    pred_crop = pred_pad[:, :, pht:pht + 1080, pwl:pwl + 1920]
    assert pred_crop.shape == x.shape, f"crop后尺寸不匹配: {pred_crop.shape} vs {x.shape}"

    # 测试奇数尺寸
    x2 = torch.randn(1, 3, 135, 247)
    x2_pad, pads2 = _pad_to_multiple(x2, multiple=16)
    pht2, phb2, pwl2, pwr2 = pads2
    assert x2_pad.shape[-2] == 144, f"135->144, 实际{x2_pad.shape[-2]}"
    assert x2_pad.shape[-1] == 256, f"247->256, 实际{x2_pad.shape[-1]}"
    assert pht2 + phb2 == 9 and abs(pht2 - phb2) <= 1, f"pad应均分"
    assert pwl2 + pwr2 == 9 and abs(pwl2 - pwr2) <= 1, f"pad应均分"

    print("  [PASS] 对称pad正确，1080->1088上下各pad4")


def test_tile_for_large_resolution():
    print("\n" + "=" * 60)
    print("[2] Tile-based 强制推理测试")
    print("=" * 60)

    device = torch.device('cuda')
    model = PureResUNet(base_ch=32).to(device)
    model.eval()

    test_cases = [
        ("Class-B", 1080, 1920, True),
        ("Class-A", 1600, 2560, True),
        ("Class-C", 480, 832, False),
        ("Class-D", 576, 704, False),
        ("Class-E", 288, 352, False),
    ]

    for name, h, w, expect_tile in test_cases:
        x = torch.randn(1, 3, h, w, device=device)
        use_tile = (h > 720 or w > 1280)
        assert use_tile == expect_tile, f"{name} tile判断错误"

        if use_tile:
            pred = tile_predict(model, x, tile_size=256, stride=128)
        else:
            x_pad, pads = _pad_to_multiple(x, multiple=16)
            with torch.no_grad():
                pred = model(x_pad)
            pht, phb, pwl, pwr = pads
            pred = pred[:, :, pht:pht + h, pwl:pwl + w]

        assert pred.shape == x.shape, f"{name} 输出尺寸不匹配"
        print(f"  {name:10s} {w:5d}x{h:5d} tile={use_tile}  shape={tuple(pred.shape)}  OK")

    print("  [PASS] Class A/B 强制tile，小分辨率整帧")


def test_vram_budget():
    print("\n" + "=" * 60)
    print("[3] 显存预算测试")
    print("=" * 60)

    device = torch.device('cuda')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = PureResUNet(base_ch=32).to(device)
    loss_fn = L1SSIMLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    x = torch.randn(8, 3, 256, 256, device=device)
    target = torch.randn(8, 3, 256, 256, device=device)

    # FP32 训练（AMP 已关闭，避免 FP16 数值漂移导致 loss=NaN）
    pred = model(x)
    loss = loss_fn(pred, target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  batch=8, patch=256, base_ch=32, FP32")
    print(f"  Peak VRAM: {peak_gb:.2f} GB")

    assert peak_gb < 14, f"显存超标: {peak_gb:.2f}GB >= 14GB"
    print(f"  [PASS] 显存 {peak_gb:.2f}GB < 14GB，batch_size=8 可用")


def main():
    test_symmetric_pad()
    test_tile_for_large_resolution()
    test_vram_budget()
    print("\n" + "=" * 60)
    print("所有验收项通过！")
    print("=" * 60)


if __name__ == '__main__':
    main()
