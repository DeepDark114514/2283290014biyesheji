import os
import sys
import time
import random
import argparse
import logging
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from config import CONFIG
from models import PureResUNet, DegFiLMResUNet
from losses import L1SSIMLoss
from datasets import build_dataloader, tile_predict
from utils import EarlyStopping, calc_psnr, set_high_priority, disable_quick_edit_tip


def parse_args():
    parser = argparse.ArgumentParser(description='A/B 统一训练')
    parser.add_argument('-m', '--model_type', type=str, default=None,
                        choices=['A', 'B'])
    parser.add_argument('--base_ch', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--unseen_qp', type=int, nargs='+', default=None)
    return parser.parse_args()


def override_config(cfg, args):
    if args.model_type is not None:
        cfg['model_type'] = args.model_type
    if args.base_ch is not None:
        cfg['base_ch'] = args.base_ch
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    if args.lr is not None:
        cfg['lr'] = args.lr
    if args.unseen_qp is not None:
        cfg['unseen_qp_list'] = args.unseen_qp
    return cfg


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# A方案 PureResUNet，B方案 DegFiLMResUNet
def build_model(cfg):
    model_type = cfg.get('model_type', 'A')
    if model_type == 'A':
        return PureResUNet(base_ch=cfg['base_ch'])
    elif model_type == 'B':
        return DegFiLMResUNet(base_ch=cfg['base_ch'])
    else:
        raise ValueError(f'Unknown model_type: {model_type}')


# 看看是否改了batch_size，我这里验证一下config
print(f"[DEBUG] loading config, model_type={CONFIG.get('model_type', 'A')}")

def build_dataloader_for_split(cfg, split):
    model_type = cfg.get('model_type', 'A')
    is_train = (split == 'train')
    if model_type == 'B':
        from datasets.multi_qp_dataset import build_multi_qp_dataloader
        if is_train:
            qp_list = cfg.get('qp_list', [22, 32, 42])
        else:
            qp_list = cfg.get('qp', 32)
        return build_multi_qp_dataloader(cfg, split, qp_list=qp_list)
    else:
        if is_train:
            # A方案训练也用混合QP，和B训练集相同
            from datasets.multi_qp_dataset import build_multi_qp_dataloader
            qp_list = cfg.get('qp_list', [22, 32, 42])
            return build_multi_qp_dataloader(cfg, split, qp_list=qp_list)
        else:
            qp = cfg.get('qp', 32)
            return build_dataloader(cfg, split, qp=qp)


def build_optimizer_scheduler(model, cfg):
    optimizer_name = cfg['optimizer']
    lr = cfg['lr']
    wd = cfg['weight_decay']

    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_name}')

    scheduler_name = cfg['scheduler']
    if scheduler_name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg['step_size'], gamma=cfg['gamma']
        )
    elif scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg['epochs']
        )
    else:
        scheduler = None

    return optimizer, scheduler


def train_one_epoch(model, dataloader, loss_fn, optimizer, scaler, device, cfg, epoch, logger):
    model.train()
    total_loss = 0.0
    total_l1 = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    grad_norm = 0.0
    pbar = tqdm(dataloader, desc='Train', leave=False)

    accum_steps = cfg.get('grad_accum_steps', 1)  # 梯度累积，显存不够时相当于大batch_size
    optimizer.zero_grad()

    track_qp = (cfg.get('model_type') == 'B')
    qp_loss_sum = {}
    qp_count = {}

    for step, batch in enumerate(pbar):
        lq = batch[0].to(device, non_blocking=True)
        hq = batch[1].to(device, non_blocking=True)
        if track_qp:
            qp_batch = batch[2]

        if cfg['amp']:
            with autocast('cuda'):
                pred = model(lq)
                if cfg.get('pred_clamp', False):
                    pred = torch.clamp(pred, 0.0, 1.0)  # 先clamp再算loss，不然模型可能输出负数或>1
                loss, l1_val, ssim_loss_val, ssim_val = loss_fn(pred, hq, return_components=True)
                loss = loss / accum_steps
        else:
            pred = model(lq)
            if cfg.get('pred_clamp', False):
                pred = torch.clamp(pred, 0.0, 1.0)
            loss, l1_val, ssim_loss_val, ssim_val = loss_fn(pred, hq, return_components=True)
            loss = loss / accum_steps

        with torch.no_grad():
            psnr_val = calc_psnr(pred.clamp(0, 1), hq)

        if cfg['amp']:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accum_steps == 0:
            if cfg['amp']:
                if cfg.get('clip_grad_norm', None):
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad_norm'])  # 梯度裁剪防爆炸，1.0够用
                scaler.step(optimizer)
                scaler.update()
            else:
                if cfg.get('clip_grad_norm', None):
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad_norm'])
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        total_l1 += l1_val
        total_ssim += ssim_val
        total_psnr += psnr_val
        pbar.set_postfix({'loss': f"{loss.item() * accum_steps:.4f}"})

        if track_qp:
            for q in qp_batch.tolist():
                qp_loss_sum[q] = qp_loss_sum.get(q, 0.0) + (loss.item() * accum_steps)
                qp_count[q] = qp_count.get(q, 0) + 1

        if step % 50 == 0:
            logger.info(f"Epoch {epoch}/{cfg['epochs']}, Batch {step+1}/{len(dataloader)}, Loss: {loss.item() * accum_steps:.4f}, L1: {l1_val:.4f}, SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.2f}")

    if (step + 1) % accum_steps != 0:
        if cfg['amp']:
            if cfg.get('clip_grad_norm', None):
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
        else:
            if cfg.get('clip_grad_norm', None):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad_norm'])
            optimizer.step()
        optimizer.zero_grad()

    n = len(dataloader)
    if track_qp and qp_count:
        qp_avg = {q: qp_loss_sum[q] / qp_count[q] for q in sorted(qp_loss_sum.keys())}
        logger.info(f"Epoch {epoch} QP-loss: {qp_avg}")

    return total_loss / n, float(grad_norm), total_l1 / n, total_ssim / n, total_psnr / n


def _pad_to_multiple(x, multiple=16):  # 下采样4次，特征图尺寸要能被16整除
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    pad_h_top = pad_h // 2
    pad_h_bottom = pad_h - pad_h_top
    pad_w_left = pad_w // 2
    pad_w_right = pad_w - pad_w_left
    if pad_h > 0 or pad_w > 0:
        x = torch.nn.functional.pad(x, (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom), mode='reflect')
    return x, (pad_h_top, pad_h_bottom, pad_w_left, pad_w_right)


@torch.no_grad()
def validate(model, dataloader, device, cfg, loss_fn, desc='Val'):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_loss = 0.0
    total_frames = 0

    for batch in tqdm(dataloader, desc=desc, leave=False):
        lq = batch[0].to(device)
        hq = batch[1].to(device)
        _, _, h, w = lq.shape

        use_tile = (h > 720 or w > 1280)  # 大图用tile，小图直接整帧推

        if use_tile:
            pred = tile_predict(model, lq, tile_size=cfg['patch_size'], stride=cfg['patch_size'] // 2)
        else:
            lq_padded, pads = _pad_to_multiple(lq, multiple=16)
            try:
                pred = model(lq_padded)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    pred = tile_predict(model, lq, tile_size=cfg['patch_size'], stride=cfg['patch_size'] // 2)  # 保险起见，万一哪张图特别大
                    loss, _, _, ssim_val = loss_fn(pred.clamp(0, 1), hq, return_components=True)
                    total_psnr += float(calc_psnr(pred.clamp(0, 1), hq))
                    total_ssim += ssim_val
                    total_loss += loss.item()
                    total_frames += 1
                    continue
                else:
                    raise
            pht, phb, pwl, pwr = pads
            pred = pred[:, :, pht:pht + h, pwl:pwl + w]

        loss, _, _, ssim_val = loss_fn(pred.clamp(0, 1), hq, return_components=True)
        total_psnr += float(calc_psnr(pred.clamp(0, 1), hq))
        total_ssim += ssim_val
        total_loss += loss.item()
        total_frames += 1

    if total_frames == 0:
        return 0.0, 0.0, 0.0
    return total_psnr / total_frames, total_ssim / total_frames, total_loss / total_frames


def save_checkpoint(model, optimizer, scheduler, epoch, best_psnr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_psnr': best_psnr,
    }
    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(state, path)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_logger(exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = os.path.join(exp_dir, f'train_{timestamp}.log')

    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def main():
    args = parse_args()
    cfg = CONFIG.copy()
    cfg = override_config(cfg, args)
    set_seed(cfg['seed'])
    set_high_priority()

    if args.resume:
        exp_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.resume)))
        if not os.path.exists(exp_dir):
            raise ValueError(f'无法从 checkpoint 路径推断实验目录: {args.resume}')
    elif args.eval_only and args.checkpoint:
        exp_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
        if not os.path.exists(exp_dir):
            exp_dir = 'logs/eval'
            os.makedirs(exp_dir, exist_ok=True)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"{cfg['model_type']}_{timestamp}"
        exp_dir = os.path.join('logs', exp_name)
        os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)

    with open(os.path.join(exp_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    logger = setup_logger(exp_dir)

    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"开始训练 方案: {cfg['model_type']}")
    logger.info(f'实验目录: {exp_dir}')
    logger.info(f'设备: {device}')

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info('cuDNN benchmark: enabled')
        logger.info(f'GPU: {torch.cuda.get_device_name(0)}')
        logger.info(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    model = build_model(cfg)
    model = model.to(device)
    n_params = count_parameters(model)
    logger.info(f'模型参数量: {n_params:,} ({n_params / 1e6:.2f}M)')

    if cfg['model_type'] == 'B':
        film_params = sum(p.numel() for n, p in model.named_parameters()
                          if 'deg_estimator' in n or 'film_' in n)
        logger.info(f'  - DegEstimator + FiLM 新增参数: {film_params:,} ({film_params / 1e6:.3f}M)')
        base_params = n_params - film_params
        pct = film_params / base_params * 100
        logger.info(f'  - 相对 A 方案基线 ({base_params / 1e6:.2f}M) 增加: {pct:.2f}%')

    if hasattr(torch, 'compile'):
        try:
            import importlib.util
            triton_available = importlib.util.find_spec('triton') is not None
            if triton_available:
                model = torch.compile(model, mode='max-autotune')
                logger.info('torch.compile: enabled (mode=max-autotune)')
            else:
                logger.info('torch.compile: skipped (triton not available on Windows)')
        except Exception as e:
            logger.info(f'torch.compile failed: {e}, fallback to eager mode')

    loss_fn = L1SSIMLoss(
        l1_weight=cfg['l1_weight'],
        ssim_weight=cfg['ssim_weight']
    )
    loss_fn = loss_fn.to(device)

    optimizer, scheduler = build_optimizer_scheduler(model, cfg)
    logger.info(f"优化器: {cfg['optimizer']}, lr={cfg['lr']}")

    train_loader = build_dataloader_for_split(cfg, 'train')
    val_loader = build_dataloader_for_split(cfg, 'val')
    logger.info(f'Train batches: {len(train_loader)}, Val sequences: {len(val_loader)}')

    early_stopper = EarlyStopping(
        patience=cfg['early_stop_patience'],
        min_delta=cfg['early_stop_min_delta'],
        mode=cfg['early_stop_mode']
    )

    start_epoch = 1
    best_psnr = -1.0
    val_qp = cfg.get('qp', 32)

    if args.eval_only:
        if not args.checkpoint or not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f'评估模式需要提供 checkpoint: {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f'加载模型: {args.checkpoint}')

        val_psnr, val_ssim, val_loss = validate(model, val_loader, device, cfg, loss_fn, desc='Val')
        logger.info(f'Val QP{val_qp} | PSNR: {val_psnr:.4f} dB, SSIM: {val_ssim:.4f}, Loss: {val_loss:.4f}')

        unseen_qps = cfg.get('unseen_qp_list', [])
        for uqp in unseen_qps:
            cfg['qp'] = uqp
            uqp_loader = build_dataloader_for_split(cfg, 'val')
            u_psnr, u_ssim, u_loss = validate(model, uqp_loader, device, cfg, loss_fn, desc=f'QP{uqp}')
            logger.info(f'Unseen QP{uqp} | PSNR: {u_psnr:.4f} dB, SSIM: {u_ssim:.4f}, Loss: {u_loss:.4f}')
        return

    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f'Checkpoint not found: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint.get('best_psnr', -1.0)
        logger.info(f'从 checkpoint 恢复: {args.resume}, epoch {start_epoch - 1}, best_psnr: {best_psnr:.4f}')

    scaler = GradScaler('cuda') if cfg['amp'] else None

    import csv
    csv_path = os.path.join(exp_dir, 'training_metrics.csv')
    if args.resume and os.path.exists(csv_path):
        logger.info(f'继续写入 CSV: {csv_path}')
    else:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_l1', 'train_ssim', 'train_psnr',
                           'grad_norm', 'val_psnr', 'val_ssim', 'val_loss',
                           'test_psnr', 'test_ssim', 'test_loss', 'lr'])

    for epoch in range(start_epoch, cfg['epochs'] + 1):
        epoch_start = time.time()
        val_psnr = None
        val_ssim = None
        val_loss = None

        train_loss, grad_norm, train_l1, train_ssim, train_psnr = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device, cfg, epoch, logger
        )
        logger.info(f"Epoch {epoch}/{cfg['epochs']} summary - Loss: {train_loss:.6f}, L1: {train_l1:.6f}, SSIM: {train_ssim:.4f}, PSNR: {train_psnr:.2f}, Grad Norm: {grad_norm:.4f}")

        if epoch % cfg['val_interval'] == 0 or epoch == cfg['epochs']:
            logger.info(f"Epoch {epoch}/{cfg['epochs']} - 验证")
            val_psnr, val_ssim, val_loss = validate(model, val_loader, device, cfg, loss_fn)
            elapsed = time.time() - epoch_start
            logger.info(f"[Epoch {epoch}/{cfg['epochs']}] Train Loss: {train_loss:.4f}, Val PSNR: {val_psnr:.4f} dB, Val SSIM: {val_ssim:.4f}, Val Loss: {val_loss:.4f}, Time: {elapsed:.1f}s")

            if cfg['early_stop']:
                should_stop = early_stopper(val_psnr)
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    early_stopper.save_best_model(model, optimizer, epoch, os.path.join(exp_dir, 'best_model.pth'))
                    logger.info(f'保存最佳模型，Val PSNR: {val_psnr:.4f} dB')

                if should_stop:
                    logger.info(f'Early stopping at epoch {epoch} (best val PSNR: {early_stopper.best_score:.4f} dB)')
                    break
            else:
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    early_stopper.save_best_model(model, optimizer, epoch, os.path.join(exp_dir, 'best_model.pth'))
                    logger.info(f'保存最佳模型，Val PSNR: {val_psnr:.4f} dB')
        else:
            elapsed = time.time() - epoch_start
            logger.info(f"[Epoch {epoch}/{cfg['epochs']}] Train Loss: {train_loss:.4f}, Time: {elapsed:.1f}s")

        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{train_l1:.6f}",
                f"{train_ssim:.4f}",
                f"{train_psnr:.4f}",
                f"{grad_norm:.4f}",
                f"{val_psnr:.4f}" if val_psnr is not None else '',
                f"{val_ssim:.4f}" if val_ssim is not None else '',
                f"{val_loss:.4f}" if val_loss is not None else '',
                '', '', '',
                f"{current_lr:.6f}"
            ])

        if epoch % cfg['save_interval'] == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_psnr, os.path.join(exp_dir, 'checkpoints', f'epoch_{epoch}.pth'))

    logger.info(f'Training finished. Best val PSNR: {best_psnr:.4f} dB')


if __name__ == '__main__':
    main()
