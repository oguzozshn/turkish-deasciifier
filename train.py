"""
Eğitim döngüsü.

    python train.py --batch-size 64

Adım 1'den farkı: doğrulamada loss değil **doğruluk** takip ediliyor, çünkü
baseline'la kıyaslanacak sayı o. Checkpoint de en iyi doğruluğa göre kaydediliyor.
"""

import argparse
import math
import os
import pickle
import sys
import time

import numpy as np
import torch

from model import Deasciifier, EncoderConfig
from turkish import IGNORE

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")


def load_bins(split):
    x = np.fromfile(os.path.join(DATA_DIR, f"{split}_x.bin"), dtype=np.uint16)
    y = np.fromfile(os.path.join(DATA_DIR, f"{split}_y.bin"), dtype=np.uint8)
    return x, y


def get_batch(x, y, block_size, batch_size, device):
    ix = torch.randint(len(x) - block_size - 1, (batch_size,))
    xb = torch.stack([torch.from_numpy(x[i : i + block_size].astype(np.int64)) for i in ix])
    yb = torch.stack([torch.from_numpy(y[i : i + block_size].astype(np.int64)) for i in ix])
    if device.startswith("cuda"):
        return xb.pin_memory().to(device, non_blocking=True), yb.pin_memory().to(device, non_blocking=True)
    return xb.to(device), yb.to(device)


@torch.no_grad()
def evaluate(model, x, y, block_size, batch_size, device, ctx):
    """Doğrulama kümesinin tamamı, ardışık pencerelerle. Rastgele örnekleme
    yapmıyoruz — sayı baseline'la kıyaslanacak, oynak olmamalı."""
    model.eval()
    n = (len(x) // block_size) * block_size
    xs = torch.from_numpy(x[:n].astype(np.int64)).view(-1, block_size)
    ys = torch.from_numpy(y[:n].astype(np.int64)).view(-1, block_size)

    total_loss, n_batch, correct, total = 0.0, 0, 0, 0
    for i in range(0, len(xs), batch_size):
        xb = xs[i : i + batch_size].to(device)
        yb = ys[i : i + batch_size].to(device)
        with ctx:
            logits, loss = model(xb, yb)
        total_loss += loss.item()
        n_batch += 1
        mask = yb != IGNORE
        pred = logits.argmax(-1)
        correct += int((pred[mask] == yb[mask]).sum())
        total += int(mask.sum())

    model.train()
    return total_loss / max(1, n_batch), 100 * correct / max(1, total)


def get_lr(it, warmup, max_iters, lr, min_lr):
    if it < warmup:
        return lr * (it + 1) / (warmup + 1)
    if it >= max_iters:
        return min_lr
    ratio = (it - warmup) / max(1, max_iters - warmup)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (lr - min_lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--n-embd", type=int, default=384)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--max-iters", type=int, default=5000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        use_bf16 = torch.cuda.is_bf16_supported()
        ctx = torch.autocast("cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16)
        print(f"cihaz: {torch.cuda.get_device_name(0)} | {'bf16' if use_bf16 else 'fp16'}")
    else:
        ctx = torch.autocast("cpu", enabled=False)
        print("cihaz: CPU")

    with open(os.path.join(DATA_DIR, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)

    train_x, train_y = load_bins("train")
    val_x, val_y = load_bins("val")
    print(f"train {len(train_x):,} token | val {len(val_x):,} token")

    cfg = EncoderConfig(
        vocab_size=meta["vocab_size"],
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = Deasciifier(cfg).to(device)
    print(f"parametre: {model.num_params() / 1e6:.2f}M | sözlük: {cfg.vocab_size}")

    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=torch.cuda.is_available(),
    )

    best_acc = 0.0
    t0 = time.time()

    for it in range(args.max_iters + 1):
        lr = get_lr(it, args.warmup, args.max_iters, args.lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if it % args.eval_interval == 0:
            val_loss, val_acc = evaluate(model, val_x, val_y, args.block_size, args.batch_size, device, ctx)
            print(f"iter {it:5d} | val loss {val_loss:.4f} | dogruluk {val_acc:.2f}% "
                  f"| lr {lr:.2e} | {(time.time() - t0) / 60:.1f} dk")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({"model": model.state_dict(), "config": cfg.__dict__,
                            "iter": it, "val_acc": best_acc}, os.path.join(ROOT, "ckpt.pt"))

        if it == args.max_iters:
            break

        xb, yb = get_batch(train_x, train_y, args.block_size, args.batch_size, device)
        with ctx:
            _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    print(f"\nBitti. En iyi doğruluk: {best_acc:.2f}% -> ckpt.pt")
    print(f"Toplam süre: {(time.time() - t0) / 60:.1f} dakika")


if __name__ == "__main__":
    main()
