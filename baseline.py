"""
İstatistiksel baseline: kelime sıklığı tablosu.

Mantık basit — eğitim metnindeki her kelimenin şapkasız halini anahtar yapıp,
o anahtarın en sık karşılık geldiği şapkalı hali sakla. "kisi" gördüğünde
"kişi" de, çünkü korpusta "kişi" "kışı"dan daha sık geçiyor.

Bu baseline şaşırtıcı derecede güçlü ve modelin yenmesi gereken çıta bu.
"Transformer eğittim" demek kolay; "transformer şu basit yöntemi şu kadar
geçti" demek bir şey ifade ediyor.

    python baseline.py
"""

import os
import pickle
import re
import sys
from collections import Counter, defaultdict

import numpy as np

from turkish import AMBIGUOUS, IGNORE, apply_labels

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")

WORD_RE = re.compile(r"[A-Za-z]+")


def load_split(split, itos):
    x = np.fromfile(os.path.join(DATA_DIR, f"{split}_x.bin"), dtype=np.uint16)
    y = np.fromfile(os.path.join(DATA_DIR, f"{split}_y.bin"), dtype=np.uint8)
    ascii_text = "".join(itos[int(i)] for i in x)
    return ascii_text, y


def build_table(ascii_text, labels):
    """şapkasız kelime -> (en sık şapkalı hali, o halin sayısı, toplam sayı)

    Sadece en sık cevabı değil güveni de saklıyoruz: "kisi" korpusta %99
    "kişi" ise tabloya güvenilir, %55/%45 bölünmüşse karar bağlama aittir.
    Hibrit yöntem bu bilgiyi kullanıyor.
    """
    counts = defaultdict(Counter)
    original = apply_labels(ascii_text, labels)
    for m in WORD_RE.finditer(ascii_text):
        a, b = m.span()
        counts[ascii_text[a:b]][original[a:b]] += 1

    table = {}
    for word, c in counts.items():
        best, n_best = c.most_common(1)[0]
        table[word] = (best, n_best, sum(c.values()))
    return table


def predict(ascii_text, table):
    """Tabloyu uygula. Bilinmeyen kelimeye dokunma."""
    out = list(ascii_text)
    for m in WORD_RE.finditer(ascii_text):
        a, b = m.span()
        entry = table.get(ascii_text[a:b])
        if entry is not None and len(entry[0]) == b - a:
            out[a:b] = entry[0]
    return "".join(out)


def score(pred_text, true_text, ascii_text):
    """İki ölçü: belirsiz karakterlerde doğruluk, ve tam kelime doğruluğu."""
    char_total = char_ok = 0
    for p, t, a in zip(pred_text, true_text, ascii_text):
        if a in AMBIGUOUS:
            char_total += 1
            char_ok += p == t

    word_total = word_ok = 0
    for m in WORD_RE.finditer(ascii_text):
        a, b = m.span()
        if any(ch in AMBIGUOUS for ch in ascii_text[a:b]):
            word_total += 1
            word_ok += pred_text[a:b] == true_text[a:b]

    return {
        "char_acc": 100 * char_ok / max(1, char_total),
        "word_acc": 100 * word_ok / max(1, word_total),
        "char_total": char_total,
        "word_total": word_total,
    }


def main():
    with open(os.path.join(DATA_DIR, "meta.pkl"), "rb") as f:
        itos = pickle.load(f)["itos"]

    print("Eğitim verisi okunuyor...")
    train_ascii, train_y = load_split("train", itos)
    print("Tablo kuruluyor...")
    table = build_table(train_ascii, train_y)
    print(f"Tabloda {len(table):,} farklı kelime var")

    val_ascii, val_y = load_split("val", itos)
    val_true = apply_labels(val_ascii, val_y)

    # Sıfır çaba: hiçbir şey yapma, ASCII olarak bırak
    naive = score(val_ascii, val_true, val_ascii)
    # Tablo
    pred = predict(val_ascii, table)
    result = score(pred, val_true, val_ascii)

    unseen = sum(
        1
        for m in WORD_RE.finditer(val_ascii)
        if m.group() not in table and any(c in AMBIGUOUS for c in m.group())
    )

    print(f"\nDoğrulama: {result['char_total']:,} belirsiz karakter, "
          f"{result['word_total']:,} belirsiz kelime")
    print(f"Tabloda olmayan kelime: {unseen:,} (%{100 * unseen / result['word_total']:.1f})")
    print("\n" + "-" * 48)
    print(f"{'yöntem':<24}{'karakter':>11}{'kelime':>12}")
    print("-" * 48)
    print(f"{'hiçbir şey yapma':<24}{naive['char_acc']:>10.2f}%{naive['word_acc']:>11.2f}%")
    print(f"{'kelime tablosu':<24}{result['char_acc']:>10.2f}%{result['word_acc']:>11.2f}%")
    print("-" * 48)

    with open(os.path.join(ROOT, "baseline.pkl"), "wb") as f:
        pickle.dump(table, f)
    print("\nTablo baseline.pkl'e yazıldı")


if __name__ == "__main__":
    main()
