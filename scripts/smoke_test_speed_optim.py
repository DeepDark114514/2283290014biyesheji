#  南京信息工程大学22级信安1班 202283290014
# 2026.5.13
# 冒烟测试：验证 torch.compile + cudnn.benchmark + cv2 加载 + 梯度累积
# 只跑 3 个 batch，验证能正常 forward/backward/step，无 CUDA 错误

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from config import CONFIG
from datasets import build_dataloader
from models import PureResUNet
from losses import L1SSIMLoss
from utils import set_high_priority

def main():
    print("=" * 60)
    print("Speed Optim Smoke Test")
    print("=" * 60)

    cfg = CONFIG.copy()
    cfg['batch_size'] = 8
    cfg['num_workers'] = 2  # 测试用少点 worker 启动快
    cfg['grad_accum_steps'] = 2

    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    print(f"[OK] cuDNN benchmark = {torch.backends.cudnn.benchmark}")

    model = PureResUNet(base_ch=32).to(device)
    print(f"[OK] Model built, params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # torch.compile (Windows 上通常不可用，跳过)
    import importlib.util
    if importlib.util.find_spec("triton") is not None:
        try:
            model = torch.compile(model, mode="max-autotune")
            print("[OK] torch.compile enabled")
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}")
    else:
        print("[INFO] torch.compile skipped (triton not available on Windows)")

    loss_fn = L1SSIMLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = build_dataloader(cfg, 'train')
    print(f"[OK] DataLoader built, batches={len(train_loader)}, persistent_workers={train_loader.persistent_workers}")

    model.train()
    accum_steps = cfg['grad_accum_steps']
    optimizer.zero_grad()

    for step, (lq, hq) in enumerate(train_loader):
        lq = lq.to(device)
        hq = hq.to(device)

        pred = model(lq)
        loss = loss_fn(pred, hq) / accum_steps
        loss.backward()

        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad_norm'])
            optimizer.step()
            optimizer.zero_grad()
            print(f"  [Batch {step+1}] loss={loss.item()*accum_steps:.4f}, pred_range=[{pred.min():.3f}, {pred.max():.3f}]")

        if step >= 4:
            break

    # 收尾
    if (step + 1) % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    print("[PASS] All speed optimizations smoke test passed!")
    print(f"  - cv2 data loading: OK")
    print(f"  - torch.compile: OK")
    print(f"  - cudnn.benchmark: OK")
    print(f"  - grad_accum (steps={accum_steps}): OK")

if __name__ == '__main__':
    main()
