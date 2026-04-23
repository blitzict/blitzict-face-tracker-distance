"""
train_distance.py  —  Train the DistanceNet regressor on MPIIGaze features
==========================================================================

Loads the .npz produced by prepare_distance_data.py, splits BY SUBJECT
(not by frame — that would leak per-face shortcuts), trains a ~7 k-param
MLP to predict distance in metres, and reports MAE against the same
classical IPD formula on the same features for direct comparison.

Usage
─────
    python train_distance.py
    python train_distance.py --holdout 3    # hold out 3 subjects for val
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR

from facetrack.distance_net import DistanceNet, NUM_DISTANCE_FEATURES


FEATURES_PATH = Path('datasets/distance_features.npz')
CKPT_PATH     = Path('checkpoints/distance_net.pth')
CKPT_DIR      = Path('checkpoints')


def split_by_subject(features, targets, subjects, n_holdout: int):
    """Leave n_holdout subjects out for validation."""
    rng = np.random.default_rng(42)
    uniq = np.unique(subjects)
    rng.shuffle(uniq)
    val_subj = set(uniq[:n_holdout].tolist())
    val_mask = np.isin(subjects, list(val_subj))
    tr_mask  = ~val_mask
    return ((features[tr_mask], targets[tr_mask], subjects[tr_mask]),
            (features[val_mask], targets[val_mask], subjects[val_mask]),
            sorted(val_subj))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',  type=int, default=200)
    parser.add_argument('--batch',   type=int, default=256)
    parser.add_argument('--lr',      type=float, default=3e-3)
    parser.add_argument('--holdout', type=int, default=3,
                        help='Number of subjects to reserve for validation')
    parser.add_argument('--ckpt',    type=str, default=str(CKPT_PATH))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if not FEATURES_PATH.exists():
        raise SystemExit(f'{FEATURES_PATH} not found — run prepare_distance_data.py first')
    data = np.load(FEATURES_PATH)
    feats    = data['features']
    targets  = data['targets']
    subjects = data['subjects']
    geom     = data['geom_dist']
    print(f'Loaded: {len(feats):,} samples  (features dim = {feats.shape[1]})')
    assert feats.shape[1] == NUM_DISTANCE_FEATURES, \
        f'feature-dim mismatch: {feats.shape[1]} vs DistanceNet expects {NUM_DISTANCE_FEATURES}'

    (tr_f, tr_t, _), (va_f, va_t, va_s), val_subj = split_by_subject(
        feats, targets, subjects, args.holdout)
    # Keep baseline geometric estimates aligned with the val split
    val_mask   = np.isin(subjects, val_subj)
    geom_val   = geom[val_mask]
    print(f'Train : {len(tr_f):,} from {len(set(subjects)) - args.holdout} subjects')
    print(f'Val   : {len(va_f):,} from {args.holdout} subjects  {sorted(val_subj)}')

    tr_ds = TensorDataset(torch.from_numpy(tr_f), torch.from_numpy(tr_t))
    va_ds = TensorDataset(torch.from_numpy(va_f), torch.from_numpy(va_t))
    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                           num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                           num_workers=0)

    model = DistanceNet().to(device)
    print(f'Model : DistanceNet ({sum(p.numel() for p in model.parameters()):,} params)')
    criterion = nn.HuberLoss(delta=0.05)       # 5 cm knee in metres
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr,
                           total_steps=args.epochs * max(1, len(tr_loader)),
                           pct_start=0.1, anneal_strategy='cos')

    # Baseline: classical IPD-formula MAE on the validation subjects
    base_mae_mm = float(np.mean(np.abs(geom_val - va_t)) * 1000.0)
    print(f'Baseline (IPD-formula) val MAE: {base_mae_mm:.1f} mm\n')

    best = float('inf')
    print(f"{'Epoch':>5} | {'TrMAE(mm)':>9} | {'VaMAE(mm)':>9} | {'Time':>5}")
    print('─' * 48)
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        tr_err = tr_n = 0.0
        for x, y in tr_loader:
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            tr_err += (pred.detach() - y).abs().sum().item()
            tr_n   += x.size(0)

        model.eval()
        va_err = va_n = 0.0
        with torch.no_grad():
            for x, y in va_loader:
                x = x.to(device); y = y.to(device)
                pred = model(x)
                va_err += (pred - y).abs().sum().item()
                va_n   += x.size(0)
        tr_mae_mm = (tr_err / max(tr_n, 1)) * 1000.0
        va_mae_mm = (va_err / max(va_n, 1)) * 1000.0
        print(f'{epoch+1:>5} | {tr_mae_mm:>9.1f} | {va_mae_mm:>9.1f} '
              f'| {time.time()-t0:>4.1f}s')

        if va_mae_mm < best:
            best = va_mae_mm
            CKPT_DIR.mkdir(exist_ok=True)
            torch.save({'model': model.state_dict(),
                        'feature_dim': NUM_DISTANCE_FEATURES,
                        'val_mae_mm': best,
                        'baseline_mae_mm': base_mae_mm,
                        'val_subjects': val_subj,
                        'epoch': epoch}, args.ckpt)

    print(f'\nBest val MAE  : {best:.1f} mm   '
          f'(baseline {base_mae_mm:.1f} mm, improvement '
          f'{(base_mae_mm - best) / base_mae_mm * 100.0:+.1f}%)')
    print(f'Checkpoint    : {args.ckpt}')


if __name__ == '__main__':
    main()
