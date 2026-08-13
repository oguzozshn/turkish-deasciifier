"""
Bağlama duyarlılık testi.

Aynı şapkasız kelime, iki farklı cümlede farklı çözülmeli. Kelime tablosu
bunu yapısı gereği yapamaz — bir anahtarın tek bir cevabı vardır, hep en sık
olanı seçer. Modelin tek gerçek üstünlüğü burası.

    python demo.py
"""

import os
import pickle
import sys

import torch

from baseline import predict as baseline_predict
from infer import predict_labels
from model import Deasciifier, EncoderConfig
from turkish import apply_labels

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.abspath(__file__))

# (şapkasız cümle, beklenen kritik kelime)
PAIRS = [
    ("bacagimda kotu bir aci var", "acı"),
    ("ucgenin ic acisi altmis derece", "açısı"),
    ("saclarimi dun kestirdim", "saçlarımı"),
    ("catinin saci yagmurdan paslanmis", "sacı"),
    ("sabah dus almadan disari cikmam", "duş"),
    ("bu gece cok tuhaf bir dus gordum", "düş"),
    ("kus kanatlarini acip havalandi", "kuş"),
    ("bu kisi toplantiya katilmadi", "kişi"),
    ("gecen kisi karda gecirdik", "kışı"),
    ("yarin sabah gorusuruz", "görüşürüz"),
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(ROOT, "data", "meta.pkl"), "rb") as f:
        stoi = pickle.load(f)["stoi"]
    with open(os.path.join(ROOT, "baseline.pkl"), "rb") as f:
        table = pickle.load(f)

    ckpt = torch.load(os.path.join(ROOT, "ckpt.pt"), map_location=device, weights_only=False)
    model = Deasciifier(EncoderConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"checkpoint: iter {ckpt['iter']}, {ckpt['val_acc']:.2f}%\n")
    hit = 0
    for text, expected in PAIRS:
        model_out = apply_labels(text, predict_labels(model, text, stoi, device))
        table_out = baseline_predict(text, table)
        ok = expected in model_out.split()
        hit += ok
        print(f"  girdi : {text}")
        print(f"  model : {model_out}   {'✓' if ok else '✗ (beklenen: ' + expected + ')'}")
        print(f"  tablo : {table_out}")
        print()

    print(f"Model {hit}/{len(PAIRS)} kritik kelimeyi doğru çözdü.")


if __name__ == "__main__":
    main()
