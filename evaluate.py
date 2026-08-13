"""
Model ile baseline'ı aynı doğrulama kümesinde, aynı pozisyonlarda kıyasla.

    python evaluate.py

Üç şey basar:
  1. Genel doğruluk tablosu (hiçbir şey yapma / kelime tablosu / model)
  2. Harf bazında kırılım — hangi harf zor
  3. İkisinin ayrıştığı örnekler — modelin gerçekten ne kazandırdığı
"""

import os
import pickle
import sys

import numpy as np
import torch

from baseline import WORD_RE, load_split, predict as baseline_predict, score
from infer import predict_labels
from model import Deasciifier, EncoderConfig
from turkish import AMBIGUOUS, apply_labels

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.abspath(__file__))


def per_letter(pred_text, true_text, ascii_text):
    stats = {}
    for p, t, a in zip(pred_text, true_text, ascii_text):
        if a in AMBIGUOUS:
            total, ok = stats.get(a, (0, 0))
            stats[a] = (total + 1, ok + (p == t))
    return stats


def disagreements(ascii_text, true_text, a_text, b_text, limit=6, seed=0):
    """a'nın doğru, b'nin yanlış olduğu kelimeleri bağlamıyla topla.

    Tamamını toplayıp rastgele örnekliyoruz. Baştan ilk N tanesini almak
    yanıltıcı: doğrulama kümesinin başı tek bir makale, örnekler o makalenin
    tuhaflıklarını yansıtıyor, geneli değil.
    """
    found = []
    for m in WORD_RE.finditer(ascii_text):
        lo, hi = m.span()
        if not any(ch in AMBIGUOUS for ch in ascii_text[lo:hi]):
            continue
        truth = true_text[lo:hi]
        if a_text[lo:hi] == truth and b_text[lo:hi] != truth:
            ctx = true_text[max(0, lo - 32) : hi + 32].replace("\n", " ")
            found.append((ascii_text[lo:hi], b_text[lo:hi], truth, ctx))

    total = len(found)
    if total > limit:
        rng = np.random.default_rng(seed)
        found = [found[i] for i in sorted(rng.choice(total, limit, replace=False))]
    return found, total


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(os.path.join(ROOT, "data", "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]

    ascii_text, labels = load_split("val", itos)
    true_text = apply_labels(ascii_text, labels)

    with open(os.path.join(ROOT, "baseline.pkl"), "rb") as f:
        table = pickle.load(f)
    base_text = baseline_predict(ascii_text, table)

    ckpt = torch.load(os.path.join(ROOT, "ckpt.pt"), map_location=device, weights_only=False)
    model = Deasciifier(EncoderConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model_text = apply_labels(ascii_text, predict_labels(model, ascii_text, stoi, device))

    naive = score(ascii_text, true_text, ascii_text)
    base = score(base_text, true_text, ascii_text)
    mdl = score(model_text, true_text, ascii_text)

    print(f"checkpoint: iter {ckpt['iter']}, egitim sirasi dogruluk {ckpt['val_acc']:.2f}%")
    print(f"dogrulama: {base['char_total']:,} belirsiz karakter, {base['word_total']:,} kelime\n")

    print("-" * 52)
    print(f"{'yöntem':<24}{'karakter':>12}{'kelime':>14}")
    print("-" * 52)
    for name, s in (("hiçbir şey yapma", naive), ("kelime tablosu", base), ("transformer", mdl)):
        print(f"{name:<24}{s['char_acc']:>11.2f}%{s['word_acc']:>13.2f}%")
    print("-" * 52)

    err_base = 100 - base["word_acc"]
    err_mdl = 100 - mdl["word_acc"]
    if err_mdl < err_base:
        print(f"\nKelime hatası {err_base:.2f}% -> {err_mdl:.2f}% "
              f"(%{100 * (err_base - err_mdl) / err_base:.0f} azalma)")

    print("\nHarf bazında (kelime tablosu / transformer):")
    sb, sm = per_letter(base_text, true_text, ascii_text), per_letter(model_text, true_text, ascii_text)
    for ch in sorted(sb, key=lambda c: -sb[c][0]):
        tb, ob = sb[ch]
        tm, om = sm[ch]
        print(f"  {ch}: {tb:>6,} kez   {100 * ob / tb:6.2f}%  /  {100 * om / tm:6.2f}%")

    picks, total = disagreements(ascii_text, true_text, model_text, base_text)
    print(f"\n--- modelin bildiği, tablonun bilmediği ({total:,} kelime, rastgele 6) ---")
    for w, got, truth, ctx in picks:
        print(f'  "{w}" -> tablo "{got}", doğru "{truth}"\n      ...{ctx}...')

    picks, total = disagreements(ascii_text, true_text, base_text, model_text)
    print(f"\n--- tablonun bildiği, modelin bilmediği ({total:,} kelime, rastgele 6) ---")
    for w, got, truth, ctx in picks:
        print(f'  "{w}" -> model "{got}", doğru "{truth}"\n      ...{ctx}...')


if __name__ == "__main__":
    main()
