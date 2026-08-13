"""
Türkçe'ye özgü harfler ve ASCII karşılıkları.

Problem şu: Türkçe metni şapkasız yazdığında bilgi kayboluyor.
"kisi" hem "kişi" hem "kışı" olabilir. "sac" hem "saç" hem "sac".
Geri koymak için bağlama bakmak gerekiyor — bu yüzden makine öğrenmesi işi.
"""

# Belirsiz ASCII harfi -> Türkçe karşılığı.
# Dikkat: küçük i -> ı (noktasız), büyük I -> İ (noktalı). Türkçe'de i/I
# çifti diğer dillerdekinden farklı çalışır, asciify de bu yönde eşler.
TO_TURKISH = {
    "c": "ç", "g": "ğ", "i": "ı", "o": "ö", "s": "ş", "u": "ü",
    "C": "Ç", "G": "Ğ", "I": "İ", "O": "Ö", "S": "Ş", "U": "Ü",
}

TO_ASCII = {tr: ascii_ch for ascii_ch, tr in TO_TURKISH.items()}

# Yalnızca bu harflerde karar vermek gerekiyor. Metnin geri kalanı zaten kesin.
AMBIGUOUS = frozenset(TO_TURKISH)

# Etiketler
KEEP = 0      # ASCII hali doğru: "sac" -> "sac"
CONVERT = 1   # Türkçe haline çevir: "sac" -> "saç"
IGNORE = 2    # belirsiz olmayan karakter; loss'a katılmaz


def asciify(text: str) -> str:
    """Türkçe metnin şapkalarını sök. Eğitim verisi bu şekilde bedava üretiliyor."""
    return "".join(TO_ASCII.get(ch, ch) for ch in text)


def make_labels(original: str, ascii_text: str) -> list[int]:
    """Her pozisyon için: KEEP, CONVERT ya da IGNORE."""
    labels = []
    for orig_ch, ascii_ch in zip(original, ascii_text):
        if ascii_ch not in AMBIGUOUS:
            labels.append(IGNORE)
        else:
            labels.append(CONVERT if orig_ch != ascii_ch else KEEP)
    return labels


def apply_labels(ascii_text: str, labels) -> str:
    """Tahmin edilen etiketleri uygulayıp metni geri kur."""
    out = []
    for ch, label in zip(ascii_text, labels):
        out.append(TO_TURKISH[ch] if label == CONVERT and ch in AMBIGUOUS else ch)
    return "".join(out)
