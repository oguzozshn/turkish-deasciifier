# turkish-deasciifier

Şapkasız yazılmış Türkçe metni düzeltir: `"cok guzel bir gun"` → `"çok güzel bir gün"`.

Üç yöntem karşılaştırılıyor — istatistiksel bir kelime tablosu, sıfırdan yazılmış
bir transformer encoder, ve ikisinin birleşimi.

Transformer bloklarının kaynağı: [nano-turkish-gpt](https://github.com/oguzozshn/nano-turkish-gpt).
Oradaki GPT ile tek yapısal fark **nedensel maskenin olmaması** — orada geleceği
görmek hileydi, burada şart. `"kisi"` kelimesinin `"kişi"` mi `"kışı"` mı olduğunu
anlamak için sağdaki eke bakmak gerekiyor.

## Sonuçlar

Doğrulama kümesinin ikinci yarısında. Eşik ayarı ilk yarıda yapıldı, bu yarıya
hiç bakılmadı.

| yöntem | karakter | kelime |
|---|---:|---:|
| hiçbir şey yapma | 66.48% | 50.49% |
| kelime tablosu | 98.22% | 97.06% |
| transformer | 99.24% | 98.43% |
| **hibrit** | **99.59%** | **99.16%** |
| *(tavan: her kelimede kusursuz seçim)* | *99.77%* | *99.59%* |

Kelime tablosuna göre hata **%2.94 → %0.84**, yani %71 azalma. Tavan %99.59;
hibrit oradaki boşluğun neredeyse tamamını kapatıyor.

## Baseline neden önce yazıldı

İlk eğitim 5000 adımdı ve **%97.03** verdi — kelime tablosunun **%97.68**'inin
altında. Yani transformer basit bir sözlüğe yenildi.

Baseline olmasaydı %97.03'ü görüp "başardık" derdik. Baseline olduğu için
kaybettiğimizi gördük, harf bazında kırılıma bakınca da *nerede* kaybettiğimizi:

| harf | örnek | tablo | transformer |
|---|---:|---:|---:|
| i | 40.833 | 97.04% | **99.29%** |
| s | 14.283 | 98.04% | **98.54%** |
| o | 9.942 | 98.75% | **99.35%** |
| g | 6.034 | 97.78% | **99.70%** |
| O | 516 | 89.15% | **99.42%** |
| S | 1.094 | **98.45%** | 96.44% |
| C | 674 | **98.07%** | 94.66% |

Model bol veri olan yerde kazanıyor, seyrek olan yerde çöküyor. Tablo ise gördüğü
kelimeyi ezberden biliyor ama görmediğine hiç dokunmuyor (doğrulamadaki kelimelerin
%4.9'u tabloda yok). İkisi **farklı yerlerde** hata yaptığı için birleştirmek mantıklı
oldu — hibrit fikri buradan çıktı, baştan tasarlanmadı.

Eğitim 15.000 adıma uzatılınca transformer tek başına da tabloyu geçti (%98.43).
İlk koşu yakınsamamıştı; doğruluk eğrisi LR planı bittiğinde hâlâ tırmanıyordu.

## Hibrit nasıl çalışıyor

Kural tek satır: **kelime tabloda ve tablo yeterince eminse tabloyu kullan, değilse modeli.**

Tablo her kelime için sadece en sık karşılığı değil, o karşılığın oranını da
saklıyor. `"kisi"` korpusta %99 `"kişi"` ise tabloya güvenilir; %55/%45 bölünmüşse
karar bağlama aittir, orası modelin işi.

Eşik taraması `min_conf = 0.90` seçti. Kelimelerin **%90.7'sinde tablo**, kalan
%9.3'ünde model kullanılıyor — modelin bütün katkısı o %9.3'ten geliyor.

## Veri: etiketleme bedava

Bu görevin en güzel tarafı: insan etiketlemesi gerekmiyor. Herhangi bir Türkçe
metnin şapkalarını sökersen hedef zaten elinde.

```
38.6M karakter (Türkçe Wikipedia)
 9.44M belirsiz pozisyon (%24.5) — karar verilecek yerler
   %33.6'sı şapkalı
```

Belirsiz olan yalnızca 12 harf: `c g i o s u` ve büyükleri. Metnin geri kalanı
kesin, o pozisyonlar loss'a hiç girmiyor.

Veriden çıkan iki dilbilimsel ayrıntı:

- Büyük **G hiç dönüşmüyor** (56.406 örnekte %0.0) — Türkçe'de "Ğ" ile başlayan
  kelime yok. Hem tablo hem model bu harfte %100.
- Büyük **I'nın %70'i "İ" oluyor** (İstanbul, İngiltere, İkinci...).
- Asıl iş küçük **i**'de: 3.9M örnek, %34'ü "ı". En sık ve en zor karar.

## Bağlam gerçekten kullanılıyor mu

Kelime tablosu bir anahtara tek cevap verir — aynı şapkasız kelimeyi iki farklı
şekilde çözemez, yapısı gereği. Model çözebiliyor mu?

```
girdi : bacagimda kotu bir aci var
model : bacağımda kötü bir acı var        ✓
girdi : ucgenin ic acisi altmis derece
model : üçgenin iç açısı altmış derece    ✓
```

Aynı `aci` dizisi, iki farklı doğru cevap. Bağlam kullanılıyor.

Ama her zaman değil — `demo.py` 10 kritik kelimenin **7'sini** doğru çözüyor.
Kaybettiği üçü aynı desende:

```
catinin saci yagmurdan paslanmis  -> "saçı" dedi, doğrusu "sacı" (metal levha)
sabah dus almadan disari cikmam   -> "düş" dedi, doğrusu "duş"
gecen kisi karda gecirdik         -> "kişi" dedi, doğrusu "kışı"
```

Üçünde de nadir okuma kaybediyor. Sebebi kısmen korpus: Wikipedia ansiklopedik
metin, "duş almak" ya da "çatı sacı" gibi gündelik kullanımlar orada neredeyse
hiç geçmiyor. Sıklık önselini bağlam yenemiyor.

## Model nerede hata yapıyor

Doğrulamada modelin kaybettiği 565 kelimeye bakınca baskın desen **alıntı kelimeler**:

```
"Aramice"      -> model "Aramiçe"
"silisik"      -> model "silişik"
"terminalidir" -> model "terminalıdir"
"konsulu"      -> model "konsulu",  doğrusu "konsülü"
```

`"terminalidir"` özellikle öğretici: model ünlü uyumunu uygulayıp `-ıdir` yazıyor,
ama "terminal" alıntı bir kelime ve Türkçe ünlü uyumunu bozuyor, doğrusu `-idir`.
Model kuralı öğrenmiş, kuralın istisnasını öğrenememiş. Tablo ise bu kelimeleri
ezberden bildiği için hata yapmıyor — hibritin neden kazandığının özeti bu.

Modelin kazandığı 1.258 kelime ise ağırlıklı olarak tabloda hiç görülmemiş,
uzun ve ekli kelimeler: `"amaçladıklarını"`, `"etimologların"`, `"modernliğe"`.

## Kurulum

```
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy datasets
```

`cu128` derlemesi RTX 50 serisi (Blackwell, sm_120) için gerekli.

## Kullanım

```
python prepare.py --hf --limit-mb 40     # veriyi indir ve etiketle
python baseline.py                        # kelime tablosunu kur ve ölç
python train.py --max-iters 15000         # transformer'ı eğit
python evaluate.py                        # tablo ve modeli kıyasla
python hybrid.py                          # hibriti kur, eşiği seç, raporla
python demo.py                            # bağlama duyarlılık testi
```

Metin düzeltmek için:

```
python restore.py "cok guzel bir gun, yarin gorusuruz"
```

Adım 1'in verisi elindeyse tekrar indirmeye gerek yok:

```
python prepare.py --from-gpt-bins ..\turkish-gpt\data
```

## Eğitim

```
10.77M parametre · 138 karakterlik sözlük · 256 karakter bağlam
15.000 adım · batch 64 · RTX 5070 Laptop, bf16
27.7 dakika · en iyi doğruluk %98.78 (iter 14500)
```

| iter | doğruluk | | iter | doğruluk |
|---:|---:|---|---:|---:|
| 0 | 53.83% | | 8.000 | 97.75% |
| 1.000 | 86.85% | | 10.000 | 98.25% |
| 2.000 | 91.76% | | 12.000 | 98.52% |
| 4.000 | 95.59% | | 14.000 | 98.71% |
| 6.000 | 97.18% | | **14.500** | **98.78%** |

Eğitim sırasındaki ölçüm ardışık pencerelerle yapılıyor; `evaluate.py` kayan
pencere kullanıp her karakteri iki taraflı bağlamla tahmin ettiği için biraz
daha yüksek çıkıyor.

## Dosyalar

| Dosya | Ne var içinde |
|---|---|
| `turkish.py` | Şapka eşlemesi, etiketleme, geri kurma |
| `prepare.py` | Metin → (ASCII girdi, etiket) çiftleri |
| `model.py` | Çift yönlü encoder + karakter başına 2 sınıflı kafa |
| `baseline.py` | Kelime sıklığı tablosu ve skorlaması |
| `train.py` | Eğitim; loss değil **doğruluk** takip ediyor |
| `infer.py` | Kayan pencere çıkarımı |
| `evaluate.py` | Tablo ve model, aynı pozisyonlarda |
| `hybrid.py` | Birleşim, eşik taraması, tavan hesabı |
| `demo.py` | Bağlama duyarlılık testi |
| `restore.py` | Komut satırından metin düzeltme |

## Doğrulama

`model.py` dört şeye karşı test edildi:

- **Başlangıç loss'u = `ln(2)`** — ikili sınıflandırmada eğitilmemiş model.
  0.7012 ölçüldü, `ln(2) = 0.6931`.
- **Çift yönlülük** — girdinin ikinci yarısı değiştirildiğinde ilk yarının
  çıktısı **değişiyor** (0.0948). Bu, [nano-turkish-gpt](https://github.com/oguzozshn/nano-turkish-gpt)'deki
  nedensellik testinin aynadaki görüntüsü: orada aynı fark tam `0.00e+00`
  olmalıydı, burada 0'dan büyük olmalı.
- **IGNORE maskesi** — belirsiz olmayan pozisyonların hedefi değiştiğinde loss
  kılını kıpırdatmıyor (`0.00e+00`), geçerli hedefler değişince oynuyor.
- **Kayan pencere** — uzunluk korunuyor, iki koşu aynı sonucu veriyor.

## Bilinen sınırlar

- **Korpus tek alan.** Wikipedia ansiklopedik metin. Gündelik dilde ("duş
  almak", "çatı sacı") performans düşüyor. Farklı alan verisiyle eğitmek en
  doğrudan iyileştirme.
- **Doğrulama kümesi bitişik bir dilim.** Metnin son %1'i, yani birkaç düzine
  makale. Makale bazında rastgele bölme daha sağlıklı olurdu. Ayrıca bu dilimin
  başında vandalize edilmiş, Kiril homoglifleri içeren bir makale var (val'in
  %0.023'ü) — skorları etkilemeyecek kadar küçük ama örnekleri baştan alırsan
  yanıltıyor, o yüzden `evaluate.py` tüm kümeden rastgele örnekliyor.
- **Sadece şapka.** Noktalama ve büyük harf restorasyonu ayrı bir problem, aynı
  iskeletle çözülebilir.
