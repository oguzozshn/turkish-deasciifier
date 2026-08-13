"""
Çıkarım: uzun metni kayan pencerelerle işle.

Neden kayan pencere? Model bir seferde block_size karakter görüyor. Metni
ardışık dilimlere bölersen dilimin ilk karakterinin solunda hiç bağlam
kalmıyor, son karakterinin sağında da öyle. Oysa bu görevde karar tam da
bağlamdan çıkıyor.

Çözüm: pencereleri yarı yarıya bindir, her pencerenin sadece ortasını kullan.
Böylece her karakter en az block_size/4 kadar iki taraflı bağlamla tahmin
edilir.
"""

import numpy as np
import torch


@torch.no_grad()
def predict_labels(model, ascii_text, stoi, device, batch_size=64):
    block = model.cfg.block_size
    stride = block // 2
    margin = (block - stride) // 2

    unk = stoi.get(" ", 0)
    ids = np.array([stoi.get(ch, unk) for ch in ascii_text], dtype=np.int64)
    n = len(ids)

    if n == 0:
        return np.zeros(0, dtype=np.int64)

    # Kısa metin: tek pencere, gerekirse boşlukla doldur
    if n <= block:
        padded = np.full(block, unk, dtype=np.int64)
        padded[:n] = ids
        x = torch.from_numpy(padded).unsqueeze(0).to(device)
        return model(x)[0].argmax(-1)[0, :n].cpu().numpy()

    starts = list(range(0, n - block + 1, stride))
    if starts[-1] + block < n:
        starts.append(n - block)

    out = np.zeros(n, dtype=np.int64)
    for i in range(0, len(starts), batch_size):
        chunk = starts[i : i + batch_size]
        x = torch.from_numpy(np.stack([ids[s : s + block] for s in chunk])).to(device)
        pred = model(x)[0].argmax(-1).cpu().numpy()

        for j, s in enumerate(chunk):
            lo = s if s == starts[0] else s + margin
            hi = s + block if s == starts[-1] else s + block - margin
            out[lo:hi] = pred[j, lo - s : hi - s]

    return out
