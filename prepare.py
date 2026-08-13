"""
Veriyi hazırla: Türkçe metin -> (ASCII girdi, etiket) çiftleri.

Etiketleme bedava: metnin şapkalarını sökünce hedef zaten elimizde.
İnsan etiketlemesi yok, bu görevin en güzel tarafı bu.

Kullanım:
    python prepare.py --from-gpt-bins C:\\ML\\turkish-gpt\\data   # adım 1'in verisi
    python prepare.py --input C:\\yol\\metin.txt
    python prepare.py --hf --limit-mb 40

Çıktı (data/):
    train_x.bin, val_x.bin   uint16  ASCII karakter indeksleri
    train_y.bin, val_y.bin   uint8   KEEP / CONVERT / IGNORE
    meta.pkl                         stoi / itos
"""

import argparse
import os
import pickle
import sys
from collections import Counter

import numpy as np

from turkish import AMBIGUOUS, asciify, make_labels

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_from_gpt_bins(path):
    """Adım 1'in (nano-turkish-gpt) hazırladığı binary'yi metne geri çöz."""
    with open(os.path.join(path, "meta.pkl"), "rb") as f:
        itos = pickle.load(f)["itos"]
    chunks = []
    for name in ("train.bin", "val.bin"):
        ids = np.fromfile(os.path.join(path, name), dtype=np.uint16)
        chunks.append("".join(itos[int(i)] for i in ids))
    return "".join(chunks)


def load_from_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_from_hf(limit_mb, config):
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("`datasets` kurulu değil: pip install datasets")

    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    budget, parts, total = limit_mb * 1024 * 1024, [], 0
    for row in ds:
        parts.append(row["text"])
        total += len(row["text"].encode("utf-8"))
        if total >= budget:
            break
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-gpt-bins", help="nano-turkish-gpt'nin data/ klasörü")
    src.add_argument("--input", help="UTF-8 metin dosyası")
    src.add_argument("--hf", action="store_true")
    ap.add_argument("--hf-config", default="20231101.tr")
    ap.add_argument("--limit-mb", type=int, default=40)
    ap.add_argument("--min-count", type=int, default=200)
    ap.add_argument("--val-frac", type=float, default=0.01)
    args = ap.parse_args()

    if args.from_gpt_bins:
        text = load_from_gpt_bins(args.from_gpt_bins)
    elif args.input:
        text = load_from_file(args.input)
    else:
        text = load_from_hf(args.limit_mb, args.hf_config)
    print(f"Ham metin: {len(text):,} karakter")

    ascii_text = asciify(text)
    labels = make_labels(text, ascii_text)

    # Sözlük ASCII metinden kuruluyor — model zaten şapkasız metin görecek
    counts = Counter(ascii_text)
    keep = sorted(ch for ch, c in counts.items() if c >= args.min_count)
    stoi = {ch: i for i, ch in enumerate(keep)}
    print(f"Sözlük: {len(stoi)} karakter ({len(counts) - len(stoi)} nadir tür atıldı)")

    # Sözlük dışı karakterleri at, etiketleri de aynı hizada tut
    xs, ys = [], []
    for ch, label in zip(ascii_text, labels):
        if ch in stoi:
            xs.append(stoi[ch])
            ys.append(label)

    x = np.array(xs, dtype=np.uint16)
    y = np.array(ys, dtype=np.uint8)
    print(f"Kodlandı: {len(x):,} token")

    n_amb = int((y != 2).sum())
    n_conv = int((y == 1).sum())
    print(
        f"\nBelirsiz pozisyon: {n_amb:,} (%{100 * n_amb / len(y):.1f})"
        f"\n  bunların %{100 * n_conv / n_amb:.1f}'i şapkalı"
    )

    # Her belirsiz harfin ne sıklıkla dönüştüğü — hangi harfin zor olduğunu gösterir
    print("\nHarf bazında dönüşüm oranı:")
    itos = {i: ch for ch, i in stoi.items()}
    for ch in sorted(AMBIGUOUS):
        if ch not in stoi:
            continue
        mask = x == stoi[ch]
        total = int(mask.sum())
        if total == 0:
            continue
        conv = int((y[mask] == 1).sum())
        print(f"  {ch}: {total:>9,} kez, %{100 * conv / total:5.1f}'i dönüşüyor")

    n_val = max(1, int(len(x) * args.val_frac))
    os.makedirs(DATA_DIR, exist_ok=True)
    x[:-n_val].tofile(os.path.join(DATA_DIR, "train_x.bin"))
    y[:-n_val].tofile(os.path.join(DATA_DIR, "train_y.bin"))
    x[-n_val:].tofile(os.path.join(DATA_DIR, "val_x.bin"))
    y[-n_val:].tofile(os.path.join(DATA_DIR, "val_y.bin"))
    with open(os.path.join(DATA_DIR, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": len(stoi), "stoi": stoi, "itos": itos}, f)

    print(f"\ntrain: {len(x) - n_val:,} token")
    print(f"val  : {n_val:,} token")


if __name__ == "__main__":
    main()
