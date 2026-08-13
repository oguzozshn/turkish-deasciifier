"""
Şapkasız Türkçe metni düzelt.

    python restore.py "cok guzel bir gun, yarin gorusuruz"
    type metin.txt | python restore.py
"""

import argparse
import os
import pickle
import sys

import torch

from infer import predict_labels
from model import Deasciifier, EncoderConfig
from turkish import apply_labels

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*", help="düzeltilecek metin (boşsa stdin okunur)")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "ckpt.pt"))
    args = ap.parse_args()

    text = " ".join(args.text) if args.text else sys.stdin.read()
    if not text.strip():
        sys.exit("Metin verilmedi.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(ROOT, "data", "meta.pkl"), "rb") as f:
        stoi = pickle.load(f)["stoi"]

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = Deasciifier(EncoderConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(apply_labels(text, predict_labels(model, text, stoi, device)))


if __name__ == "__main__":
    main()
