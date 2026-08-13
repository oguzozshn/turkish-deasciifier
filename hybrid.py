"""
Hibrit: kelime tablosu + transformer.

Harf bazında kırılım şunu söylüyordu — model bol veri olan yerde (küçük i, g)
tabloyu geçiyor, seyrek olan yerde (büyük C, U, S) çöküyor. Tablo ise
gördüğü kelimeyi ezberden biliyor ama görmediğine hiç dokunmuyor.
İkisi farklı yerlerde hata yapıyorsa birleştirmek kazandırır.

Kural: kelime tabloda VE tablo yeterince eminse tabloyu kullan, yoksa modeli.

Eşik seçimi doğrulama kümesinin ilk yarısında yapılıyor, sonuç ikinci
yarısında raporlanıyor. Aynı veride hem seçip hem raporlamak, olduğundan
iyi görünen bir sayı üretirdi.

    python hybrid.py --ckpt ckpt.pt
"""

import argparse
import os
import pickle
import sys

import torch

from baseline import WORD_RE, load_split, predict as baseline_predict, score
from infer import predict_labels
from model import Deasciifier, EncoderConfig
from turkish import AMBIGUOUS, apply_labels

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.abspath(__file__))


def hybrid_predict(ascii_text, table, model_text, min_conf, min_count):
    """Tabloya güveniliyorsa tablo, değilse model."""
    out = list(model_text)
    used_table = 0
    for m in WORD_RE.finditer(ascii_text):
        a, b = m.span()
        word = ascii_text[a:b]
        # Belirsiz harf içermeyen kelimede karar verilecek bir şey yok
        if not any(ch in AMBIGUOUS for ch in word):
            continue
        entry = table.get(word)
        if entry is None:
            continue
        best, n_best, n_total = entry
        if len(best) != b - a:
            continue
        if n_total >= min_count and n_best / n_total >= min_conf:
            out[a:b] = best
            used_table += 1
    return "".join(out), used_table


def oracle(ascii_text, true_text, a_text, b_text):
    """İki yöntemden her kelimede doğru olanı seçebilseydik tavan ne olurdu."""
    out = list(a_text)
    for m in WORD_RE.finditer(ascii_text):
        lo, hi = m.span()
        truth = true_text[lo:hi]
        if a_text[lo:hi] != truth and b_text[lo:hi] == truth:
            out[lo:hi] = truth
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "ckpt.pt"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(ROOT, "data", "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]

    ascii_text, labels = load_split("val", itos)
    true_text = apply_labels(ascii_text, labels)

    with open(os.path.join(ROOT, "baseline.pkl"), "rb") as f:
        table = pickle.load(f)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = Deasciifier(EncoderConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"checkpoint: {os.path.basename(args.ckpt)}, iter {ckpt['iter']}, {ckpt['val_acc']:.2f}%\n")

    model_text = apply_labels(ascii_text, predict_labels(model, ascii_text, stoi, device))
    base_text = baseline_predict(ascii_text, table)

    # Ayarlama / raporlama bölünmesi
    mid = len(ascii_text) // 2
    dev = slice(0, mid)
    test = slice(mid, None)

    print("Eşik taraması (doğrulamanın ilk yarısı), kelime doğruluğu:")
    counts = (1, 3, 10)
    print("  min_conf " + "".join(f"{'n>=' + str(c):>10}" for c in counts))
    best_cfg, best_acc = None, -1.0
    for min_conf in (0.50, 0.70, 0.80, 0.90, 0.95, 0.99, 1.01):
        row = []
        for min_count in counts:
            pred, _ = hybrid_predict(ascii_text, table, model_text, min_conf, min_count)
            acc = score(pred[dev], true_text[dev], ascii_text[dev])["word_acc"]
            row.append(acc)
            if acc > best_acc:
                best_acc, best_cfg = acc, (min_conf, min_count)
        print(f"  {min_conf:>8.2f} " + "".join(f"{a:>9.2f}%" for a in row))
    min_conf, min_count = best_cfg
    print(f"\nSeçilen: min_conf={min_conf}, min_count={min_count} (dev {best_acc:.2f}%)")

    hyb_text, used = hybrid_predict(ascii_text, table, model_text, min_conf, min_count)
    n_words = sum(
        1 for m in WORD_RE.finditer(ascii_text) if any(c in AMBIGUOUS for c in m.group())
    )

    print("\n" + "=" * 54)
    print("DOĞRULAMANIN İKİNCİ YARISI (raporlanan sonuç)")
    print("=" * 54)
    print(f"{'yöntem':<26}{'karakter':>12}{'kelime':>14}")
    print("-" * 54)
    rows = [
        ("hiçbir şey yapma", ascii_text),
        ("kelime tablosu", base_text),
        ("transformer", model_text),
        ("hibrit", hyb_text),
    ]
    results = {}
    for name, text in rows:
        s = score(text[test], true_text[test], ascii_text[test])
        results[name] = s
        print(f"{name:<26}{s['char_acc']:>11.2f}%{s['word_acc']:>13.2f}%")

    orc = score(
        oracle(ascii_text, true_text, base_text, model_text)[test],
        true_text[test], ascii_text[test],
    )
    print(f"{'(tavan: kusursuz seçim)':<26}{orc['char_acc']:>11.2f}%{orc['word_acc']:>13.2f}%")
    print("-" * 54)

    base_err = 100 - results["kelime tablosu"]["word_acc"]
    hyb_err = 100 - results["hibrit"]["word_acc"]
    print(f"\nTabloya göre kelime hatası: {base_err:.2f}% -> {hyb_err:.2f}%", end="")
    if base_err > 0:
        print(f"  (%{100 * (base_err - hyb_err) / base_err:.0f} azalma)")
    else:
        print()
    print(f"Kelimelerin %{100 * used / n_words:.1f}'inde tablo, kalanında model kullanıldı")


if __name__ == "__main__":
    main()
