# -*- coding: utf-8 -*-
"""
===============================================================================
ISE-216 Veri Bilimi İçin İstatistik - Proje Analiz Dosyası
===============================================================================
Veri Seti  : Sephora Ürün Verisi (veri.csv)
Kaynak     : Kaggle
Açıklama   : Bu script, Sephora kozmetik ürün veri seti üzerinde kapsamlı bir
             istatistiksel analiz gerçekleştirir. Veri temizliği, dağılım
             analizi, normallik testleri ve 4 farklı hipotez testi içerir.
===============================================================================
"""

# ============================================================================
# KÜTÜPHANE İMPORTLARI
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI penceresi açılmadan grafik kaydetmek için
import seaborn as sns
from scipy import stats
import os
import warnings

# Gereksiz uyarıları kapat (grafiklerde Türkçe karakter uyarıları vb.)
warnings.filterwarnings('ignore')

# Grafik stilini ayarla
sns.set_style("whitegrid")
sns.set_palette("deep")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

# ============================================================================
# DOSYA YOLU AYARI
# ============================================================================
# Script'in bulunduğu dizine geçiş yaparak dosya yolu sorunlarını önle
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
except Exception:
    pass

# Grafiklerin kaydedileceği klasörü oluştur
os.makedirs('grafikler', exist_ok=True)

print("=" * 80)
print("ISE-216 VERİ BİLİMİ İÇİN İSTATİSTİK - PROJE ANALİZ RAPORU")
print("=" * 80)

# ============================================================================
# BÖLÜM 1: VERİ YÜKLEME VE TANIMLAMA
# ============================================================================
print("\n" + "=" * 80)
print("BÖLÜM 1: VERİ YÜKLEME VE TANIMLAMA")
print("=" * 80)

# CSV dosyasını oku
# 62. satırı bu şekilde düzelt:
df_raw = pd.read_excel('product_info.xlsx')
print(f"\n✅ Veri başarıyla yüklendi!")
print(f"\n📊 Veri Seti Boyutu:")
print(f"   - Satır Sayısı : {df_raw.shape[0]}")
print(f"   - Sütun Sayısı : {df_raw.shape[1]}")

# Her sütunun veri tipini raporla
print(f"\n📋 Sütun Adları ve Veri Tipleri:")
print("-" * 50)
for col in df_raw.columns:
    print(f"   {col:<30} → {df_raw[col].dtype}")

# Temel istatistikleri göster
print(f"\n📈 Sayısal Değişkenlerin Özet İstatistikleri:")
print(df_raw[['loves_count', 'rating', 'price_usd', 'reviews', 'child_count']].describe().round(2).to_string())

# ============================================================================
# BÖLÜM 2: EKSİK VERİ TESPİTİ VE DOLDURMA
# ============================================================================
print("\n" + "=" * 80)
print("BÖLÜM 2: EKSİK VERİ TESPİTİ VE DOLDURMA")
print("=" * 80)

# Eksik veri sayılarını hesapla
eksik_veriler = df_raw.isnull().sum()
eksik_olan = eksik_veriler[eksik_veriler > 0]

print(f"\n🔍 Eksik Veri Tespiti:")
if len(eksik_olan) > 0:
    print("-" * 50)
    for col, count in eksik_olan.items():
        oran = (count / len(df_raw)) * 100
        print(f"   {col:<30} → {count:>5} eksik ({oran:.2f}%)")
else:
    print("   Hiçbir sütunda eksik veri bulunamadı.")

# Veriyi kopyala (ham veriyi korumak için)
df = df_raw.copy()

# -------------------------------------------------------------------------
# EKSİK VERİ DOLDURMA STRATEJİSİ:
# -------------------------------------------------------------------------
# Sayısal Veriler → MEDIAN (Ortanca) ile doldur
#   Neden? Median, aykırı değerlerden (outliers) etkilenmez.
#   Örneğin, price_usd'de 5$-500$ arası ürünler var. Ortalama (mean)
#   yüksek fiyatlı ürünlerden etkilenirken, median bunlardan BAĞIMSIZDIR.
#   Bu, verimizin genel dağılımını bozmadan eksik verileri doldurmamızı sağlar.
#
# Kategorik Veriler → MODE (En sık tekrarlanan değer) ile doldur
#   Neden? Kategorik verilerde ortalama/medyan anlamsızdır.
#   Mod, en sık görülen kategoriyi temsil eder ve veri dağılımını
#   en az bozacak şekilde eksik verileri doldurur.
# -------------------------------------------------------------------------

# Sayısal sütunları median ile doldur
sayisal_sutunlar = df.select_dtypes(include=[np.number]).columns.tolist()
for col in sayisal_sutunlar:
    if df[col].isnull().sum() > 0:
        median_deger = df[col].median()
        df[col].fillna(median_deger, inplace=True)
        print(f"   ✅ '{col}' → median ({median_deger:.2f}) ile dolduruldu")

# Kategorik sütunları mode ile doldur
kategorik_sutunlar = df.select_dtypes(include=['object']).columns.tolist()
for col in kategorik_sutunlar:
    if df[col].isnull().sum() > 0:
        mode_deger = df[col].mode()[0]
        df[col].fillna(mode_deger, inplace=True)
        print(f"   ✅ '{col}' → mode ('{mode_deger}') ile dolduruldu")

# Doldurma sonrası kontrol
kalan_eksik = df.isnull().sum().sum()
print(f"\n🎯 Doldurma sonrası toplam eksik veri: {kalan_eksik}")

# ============================================================================
# BÖLÜM 3: AYKIRI DEĞER TEMİZLİĞİ (IQR YÖNTEMİ)
# ============================================================================
print("\n" + "=" * 80)
print("BÖLÜM 3: AYKIRI DEĞER TEMİZLİĞİ (IQR YÖNTEMİ)")
print("=" * 80)

# -------------------------------------------------------------------------
# IQR (Interquartile Range / Çeyrekler Arası Açıklık) Yöntemi:
#   Q1 = %25'lik değer (1. çeyrek)
#   Q3 = %75'lik değer (3. çeyrek)
#   IQR = Q3 - Q1
#   Alt Sınır = Q1 - 1.5 * IQR
#   Üst Sınır = Q3 + 1.5 * IQR
#   Bu sınırların dışında kalan değerler "aykırı" kabul edilip çıkarılır.
#
# Neden IQR? Ortalama ve standart sapma temelli yöntemler (z-skoru gibi)
# normal dağılım varsayar. IQR ise dağılımdan bağımsız çalışır ve
# aykırı değerlerin kendisinden de etkilenmez.
# -------------------------------------------------------------------------

# Temizlenecek değişkenler
temizlenecek_degiskenler = ['loves_count', 'rating', 'price_usd']

# Temizlik öncesi veri boyutu
onceki_boyut = len(df)
print(f"\n📊 Temizlik ÖNCESİ veri boyutu: {onceki_boyut} satır")

# Her değişken için IQR temizliği uygula
print(f"\n🔧 IQR Temizliği Detayları:")
print("-" * 70)
print(f"   {'Değişken':<20} {'Q1':>10} {'Q3':>10} {'IQR':>10} {'Alt Sınır':>12} {'Üst Sınır':>12}")
print("-" * 70)

# Temizlik öncesi verileri histogram/boxplot için sakla
df_oncesi = df.copy()
# --- BU BLOĞU df_oncesi = df.copy() SATIRININ ÜZERİNE YAPIŞTIR ---
for col in ['loves_count', 'rating', 'price_usd']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Şimdi kopyayı al (artık veriler sayı formatında)
df_oncesi = df.copy()
for col in temizlenecek_degiskenler:
 
    
    Q1 = df[col].quantile(0.25)
    # ... (kodun geri kalanı aynı)
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    alt_sinir = Q1 - 1.5 * IQR
    ust_sinir = Q3 + 1.5 * IQR

    print(f"   {col:<20} {Q1:>10.2f} {Q3:>10.2f} {IQR:>10.2f} {alt_sinir:>12.2f} {ust_sinir:>12.2f}")

    # Aykırı değerleri filtrele
    aykiri_sayisi = ((df[col] < alt_sinir) | (df[col] > ust_sinir)).sum()
    df = df[(df[col] >= alt_sinir) & (df[col] <= ust_sinir)]
    print(f"   → '{col}' sütunundan {aykiri_sayisi} aykırı değer çıkarıldı.")

# Temizlik sonrası veri boyutu
sonraki_boyut = len(df)
print(f"\n📊 Temizlik SONRASI veri boyutu: {sonraki_boyut} satır")

# Temizlik öncesi/sonrası karşılaştırma tablosu
print(f"\n{'='*60}")
print(f" AYKIRI DEĞER TEMİZLİĞİ ÖZET TABLOSU")
print(f"{'='*60}")
print(f"   {'Metrik':<35} {'Değer':>15}")
print(f"   {'-'*50}")
print(f"   {'Temizlik Öncesi Satır Sayısı':<35} {onceki_boyut:>15,}")
print(f"   {'Temizlik Sonrası Satır Sayısı':<35} {sonraki_boyut:>15,}")
print(f"   {'Çıkarılan Toplam Satır':<35} {onceki_boyut - sonraki_boyut:>15,}")
print(f"   {'Veri Kaybı Oranı':<35} {((onceki_boyut - sonraki_boyut) / onceki_boyut * 100):>14.2f}%")
print(f"{'='*60}")

# ============================================================================
# BÖLÜM 4: DAĞILIM ANALİZİ VE NORMALLİK TESTİ
# ============================================================================
print("\n" + "=" * 80)
print("BÖLÜM 4: DAĞILIM ANALİZİ VE NORMALLİK TESTİ (Shapiro-Wilk)")
print("=" * 80)

# -------------------------------------------------------------------------
# Shapiro-Wilk Testi:
#   H₀ (Null Hipotez)       : Veri normal dağılıma uyar.
#   H₁ (Alternatif Hipotez) : Veri normal dağılıma UYMAZ.
#   Karar Kriteri            : p < 0.05 → H₀ reddedilir (normal değil)
#                              p >= 0.05 → H₀ reddedilemez (normal kabul)
#
# NOT: Shapiro-Wilk testi en fazla 5000 gözlem ile çalışır. Büyük veri
# setlerinde rastgele örneklem alınır.
# -------------------------------------------------------------------------

analiz_degiskenleri = ['loves_count', 'rating', 'price_usd']

print(f"\n📊 Shapiro-Wilk Normallik Testi Sonuçları:")
print("-" * 70)
print(f"   {'Değişken':<20} {'Test İstatistiği':>18} {'p-değeri':>15} {'Sonuç':>15}")
print("-" * 70)

for col in analiz_degiskenleri:
    # Örneklem boyutu 5000'den büyükse rastgele örneklem al
    data = df[col].dropna()
    if len(data) > 5000:
        data_sample = data.sample(n=5000, random_state=42)
    else:
        data_sample = data

    stat, p_val = stats.shapiro(data_sample)

    if p_val < 0.05:
        sonuc = "Normal DEĞİL ❌"
    else:
        sonuc = "Normal ✅"

    print(f"   {col:<20} {stat:>18.6f} {p_val:>15.6f} {sonuc:>15}")

print(f"\n📝 Yorum: p < 0.05 olan değişkenler normal dağılıma uymamaktadır.")
print(f"   Bu durumda non-parametrik testler tercih edilebilir. Ancak ANOVA ve")
print(f"   t-testi, büyük örneklemlerde (n>30) Merkezi Limit Teoremine göre")
print(f"   sağlam (robust) sonuçlar vermektedir.")

# ============================================================================
# BÖLÜM 5: GÖRSELLEŞTİRME (Temizlik Öncesi/Sonrası Karşılaştırma)
# ============================================================================
print("\n" + "=" * 80)
print("BÖLÜM 5: GÖRSELLEŞTİRME")
print("=" * 80)

# -------------------------------------------------------------------------
# Histogram: Verinin frekans dağılımını gösterir. Çarpıklığı (skewness),
#   tepe noktasını ve genel şekli görsel olarak anlamak için kullanılır.
#
# Box-Plot (Kutu Grafiği): Medyanı, çeyreklikleri (Q1, Q3) ve aykırı
#   değerleri tek bir grafikte özetler. Verinin yayılımını ve
#   simetrisini değerlendirmek için idealdir.
# -------------------------------------------------------------------------

for col in analiz_degiskenleri:


    
    Q1 = df[col].quantile(0.25)
    # ... (kodun geri kalanı aynı)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"'{col}' Değişkeni - Temizlik Öncesi vs Sonrası Karşılaştırma",
                 fontsize=14, fontweight='bold')

    # --- Temizlik ÖNCESİ Histogram ---
    axes[0, 0].hist(df_oncesi[col].dropna(), bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Temizlik ÖNCESİ - Histogram', fontweight='bold')
    axes[0, 0].set_xlabel(col)
    axes[0, 0].set_ylabel('Frekans')
    axes[0, 0].axvline(df_oncesi[col].median(), color='black', linestyle='--', label=f'Medyan: {df_oncesi[col].median():.2f}')
    axes[0, 0].legend()

    # --- Temizlik SONRASI Histogram ---
    axes[0, 1].hist(df[col].dropna(), bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Temizlik SONRASI - Histogram', fontweight='bold')
    axes[0, 1].set_xlabel(col)
    axes[0, 1].set_ylabel('Frekans')
    axes[0, 1].axvline(df[col].median(), color='black', linestyle='--', label=f'Medyan: {df[col].median():.2f}')
    axes[0, 1].legend()

    # --- Temizlik ÖNCESİ Box-Plot ---
    axes[1, 0].boxplot(df_oncesi[col].dropna(), vert=True, patch_artist=True,
                       boxprops=dict(facecolor='#e74c3c', alpha=0.7))
    axes[1, 0].set_title('Temizlik ÖNCESİ - Box-Plot', fontweight='bold')
    axes[1, 0].set_ylabel(col)

    # --- Temizlik SONRASI Box-Plot ---
    axes[1, 1].boxplot(df[col].dropna(), vert=True, patch_artist=True,
                       boxprops=dict(facecolor='#2ecc71', alpha=0.7))
    axes[1, 1].set_title('Temizlik SONRASI - Box-Plot', fontweight='bold')
    axes[1, 1].set_ylabel(col)

    plt.tight_layout()
    dosya_adi = f'grafikler/{col}_oncesi_sonrasi.png'
    plt.savefig(dosya_adi, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Grafik kaydedildi: {dosya_adi}")

print(f"\n📝 Histogram Açıklaması: Histogram, verinin frekans dağılımını görselleştirir.")
print(f"   Çarpıklığı (sağa/sola) ve tepe noktalarını (modları) gözlemlememizi sağlar.")
print(f"   Temizlik öncesi/sonrası karşılaştırmada aykırı değerlerin etkisi net görülür.")
print(f"\n📝 Box-Plot Açıklaması: Box-Plot, medyanı çizgi ile, Q1-Q3 aralığını kutu ile,")
print(f"   ve aykırı değerleri nokta olarak gösterir. Verinin yayılımı ve simetrisi")
print(f"   hakkında kompakt bilgi sağlar.")

# ============================================================================
# BÖLÜM 6: HİPOTEZ TESTLERİ
# ============================================================================
print("\n" + "=" * 80)
print("BÖLÜM 6: HİPOTEZ TESTLERİ")
print("=" * 80)

# ============================================================================
# HİPOTEZ 1: ANOVA TESTİ
# "Luxury", "Niche" ve "Mass Market" segmentindeki markaların ortalama
# loves_count (beğeni sayıları) arasında anlamlı bir fark var mıdır?
# ============================================================================
print("\n" + "-" * 80)
print("HİPOTEZ 1: ANOVA TESTİ (Fiyat Segmentine Göre Beğeni Farkı)")
print("-" * 80)

# -------------------------------------------------------------------------
# Markaları fiyat segmentlerine göre gruplandırma:
#   - Luxury (Lüks)       : Fiyatı üst %25 diliminin üzerinde olan ürünler
#   - Mass Market (Kitle) : Fiyatı alt %25 diliminin altında olan ürünler
#   - Niche (Niş)         : Aradaki fiyat dilimindeki ürünler
#
# H₀ : μ_Luxury = μ_Niche = μ_MassMarket (Gruplar arası fark yoktur)
# H₁ : En az iki grubun ortalaması farklıdır
# α  : 0.05
# -------------------------------------------------------------------------

# Fiyat çeyrekliklerini hesapla
fiyat_q1 = df['price_usd'].quantile(0.25)
fiyat_q3 = df['price_usd'].quantile(0.75)

# Segmentleme fonksiyonu
def fiyat_segmenti_belirle(fiyat):
    """Fiyata göre pazar segmentini belirler."""
    if fiyat <= fiyat_q1:
        return 'Mass Market'
    elif fiyat >= fiyat_q3:
        return 'Luxury'
    else:
        return 'Niche'

# Segmentleri oluştur
df['fiyat_segmenti'] = df['price_usd'].apply(fiyat_segmenti_belirle)

# Grup istatistiklerini göster
print(f"\n📊 Segment Bazlı Özet İstatistikler:")
print(f"   Fiyat Eşik Değerleri: Q1 = ${fiyat_q1:.2f}, Q3 = ${fiyat_q3:.2f}")
print("-" * 60)
segment_ozet = df.groupby('fiyat_segmenti')['loves_count'].agg(['count', 'mean', 'median', 'std']).round(2)
segment_ozet.columns = ['Gözlem Sayısı', 'Ortalama', 'Medyan', 'Std. Sapma']
print(segment_ozet.to_string())

# ANOVA testi uygula
luxury_loves = df[df['fiyat_segmenti'] == 'Luxury']['loves_count']
niche_loves = df[df['fiyat_segmenti'] == 'Niche']['loves_count']
mass_loves = df[df['fiyat_segmenti'] == 'Mass Market']['loves_count']

f_stat, p_val_anova = stats.f_oneway(luxury_loves, niche_loves, mass_loves)

print(f"\n📊 ANOVA Test Sonuçları:")
print(f"   F-istatistiği : {f_stat:.4f}")
print(f"   p-değeri      : {p_val_anova:.6f}")

# Karar
if p_val_anova < 0.05:
    print(f"   Karar         : p < 0.05 → H₀ REDDEDİLDİ ❌")
    print(f"\n📌 HOCAYA SUNUM NOTU (Hipotez 1 - ANOVA):")
    print(f"   Tek yönlü ANOVA testi sonucunda F({len(luxury_loves)+len(niche_loves)+len(mass_loves)-3}) = {f_stat:.4f},")
    print(f"   p = {p_val_anova:.6f} < 0.05 bulunmuştur. Bu sonuç istatistiksel olarak anlamlıdır.")
    print(f"   H₀ hipotezi reddedilmiş olup, Luxury, Niche ve Mass Market segmentlerindeki")
    print(f"   ürünlerin ortalama beğeni sayıları arasında anlamlı bir fark olduğu sonucuna")
    print(f"   varılmıştır. Bu fark, tüketici davranışlarının fiyat segmentine göre farklılık")
    print(f"   gösterdiğini ve belirli segmentlerin daha yüksek popülariteye sahip olduğunu")
    print(f"   ortaya koymaktadır.")
else:
    print(f"   Karar         : p >= 0.05 → H₀ REDDEDİLEMEDİ ✅")
    print(f"\n📌 HOCAYA SUNUM NOTU (Hipotez 1 - ANOVA):")
    print(f"   ANOVA testi sonucunda p = {p_val_anova:.6f} >= 0.05 bulunmuştur.")
    print(f"   H₀ hipotezi reddedilememiş olup, fiyat segmentleri arasında loves_count")
    print(f"   açısından istatistiksel olarak anlamlı bir fark tespit edilememiştir.")

# Hipotez 1 Grafiği
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Hipotez 1: Fiyat Segmentine Göre Beğeni Sayısı (ANOVA)', fontsize=14, fontweight='bold')

# Box-Plot
segment_order = ['Mass Market', 'Niche', 'Luxury']
colors_h1 = ['#3498db', '#e67e22', '#e74c3c']
sns.boxplot(x='fiyat_segmenti', y='loves_count', data=df, order=segment_order,
            palette=colors_h1, ax=axes[0])
axes[0].set_title('Box-Plot: Segment Bazlı Beğeni Dağılımı', fontweight='bold')
axes[0].set_xlabel('Fiyat Segmenti')
axes[0].set_ylabel('Beğeni Sayısı (loves_count)')

# Bar-Plot (Ortalamalar)
segment_means = df.groupby('fiyat_segmenti')['loves_count'].mean().reindex(segment_order)
bars = axes[1].bar(segment_order, segment_means, color=colors_h1, edgecolor='black', alpha=0.8)
axes[1].set_title('Ortalama Beğeni Sayısı (Segment Bazlı)', fontweight='bold')
axes[1].set_xlabel('Fiyat Segmenti')
axes[1].set_ylabel('Ortalama Beğeni (loves_count)')
# Barların üstüne değer yaz
for bar_item, val in zip(bars, segment_means):
    axes[1].text(bar_item.get_x() + bar_item.get_width()/2., bar_item.get_height() + 100,
                f'{val:,.0f}', ha='center', va='bottom', fontweight='bold')

# p-değerini grafiğe ekle
fig.text(0.5, 0.01, f'ANOVA: F = {f_stat:.4f}, p = {p_val_anova:.6f}', ha='center',
         fontsize=12, style='italic',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('grafikler/hipotez1_anova.png', bbox_inches='tight')
plt.close()
print(f"\n   ✅ Grafik kaydedildi: grafikler/hipotez1_anova.png")

# ============================================================================
# HİPOTEZ 2: BAĞIMSIZ ÖRNEKLEM T-TESTİ
# variation_count (child_count) değeri 5'ten fazla olan "Çok Çeşitli"
# ürünlerin loves_count ortalaması, "Tekil" ürünlerden yüksek midir?
# ============================================================================
print("\n" + "-" * 80)
print("HİPOTEZ 2: BAĞIMSIZ ÖRNEKLEM T-TESTİ (Ürün Çeşitliliği Etkisi)")
print("-" * 80)

# -------------------------------------------------------------------------
# H₀ : μ_ÇokÇeşitli = μ_Tekil (İki grubun ortalama beğenisi eşittir)
# H₁ : μ_ÇokÇeşitli > μ_Tekil (Çok Çeşitli ürünler daha çok beğenilir)
# α  : 0.05
# Test Türü: Tek yönlü (one-tailed) bağımsız örneklem t-testi
#
# child_count: Ürünün alt varyasyon sayısı (renk, boyut vb.)
# child_count > 5 → "Çok Çeşitli"
# child_count <= 5 → "Tekil"
# -------------------------------------------------------------------------

# Grupları oluştur
df['cesitlilik_grubu'] = df['child_count'].apply(
    lambda x: 'Çok Çeşitli (>5)' if x > 5 else 'Tekil (≤5)'
)

# Grup istatistikleri
print(f"\n📊 Grup Bazlı Özet İstatistikler:")
print("-" * 60)
cesitlilik_ozet = df.groupby('cesitlilik_grubu')['loves_count'].agg(['count', 'mean', 'median', 'std']).round(2)
cesitlilik_ozet.columns = ['Gözlem Sayısı', 'Ortalama', 'Medyan', 'Std. Sapma']
print(cesitlilik_ozet.to_string())

# Grupları ayır
cok_cesitli = df[df['cesitlilik_grubu'] == 'Çok Çeşitli (>5)']['loves_count']
tekil = df[df['cesitlilik_grubu'] == 'Tekil (≤5)']['loves_count']

# Bağımsız örneklem t-testi (tek yönlü)
t_stat, p_val_ttest = stats.ttest_ind(cok_cesitli, tekil, equal_var=False)
# Tek yönlü test için p-değerini 2'ye böl (H₁: μ1 > μ2 yönünde)
p_val_tek_yonlu = p_val_ttest / 2

print(f"\n📊 t-Testi Sonuçları:")
print(f"   t-istatistiği : {t_stat:.4f}")
print(f"   p-değeri (çift yönlü) : {p_val_ttest:.6f}")
print(f"   p-değeri (tek yönlü)  : {p_val_tek_yonlu:.6f}")

if p_val_tek_yonlu < 0.05 and t_stat > 0:
    print(f"   Karar         : p < 0.05 ve t > 0 → H₀ REDDEDİLDİ ❌")
    print(f"\n📌 HOCAYA SUNUM NOTU (Hipotez 2 - t-Testi):")
    print(f"   Bağımsız örneklem t-testi sonucunda t = {t_stat:.4f},")
    print(f"   p(tek yönlü) = {p_val_tek_yonlu:.6f} < 0.05 bulunmuştur.")
    print(f"   H₀ hipotezi reddedilmiştir. Varyasyon sayısı 5'ten fazla olan 'Çok Çeşitli'")
    print(f"   ürünlerin ortalama beğeni sayısı ({cok_cesitli.mean():,.0f}), 'Tekil' ürünlerden")
    print(f"   ({tekil.mean():,.0f}) istatistiksel olarak anlamlı derecede yüksektir.")
    print(f"   Bu durum, tüketicilere daha fazla seçenek sunmanın (renk, boyut vb.)")
    print(f"   ürün popülaritesini artırdığını göstermektedir.")
else:
    print(f"   Karar         : H₀ REDDEDİLEMEDİ ✅")
    print(f"\n📌 HOCAYA SUNUM NOTU (Hipotez 2 - t-Testi):")
    print(f"   t-testi sonucunda p(tek yönlü) = {p_val_tek_yonlu:.6f} >= 0.05 bulunmuştur.")
    print(f"   H₀ hipotezi reddedilememiştir. Ürün çeşitliliğinin beğeni sayısı üzerinde")
    print(f"   istatistiksel olarak anlamlı bir etkisi tespit edilememiştir.")

# Hipotez 2 Grafiği
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Hipotez 2: Ürün Çeşitliliğinin Beğeni Üzerindeki Etkisi (t-Testi)', fontsize=14, fontweight='bold')

# Box-Plot
colors_h2 = ['#9b59b6', '#1abc9c']
sns.boxplot(x='cesitlilik_grubu', y='loves_count', data=df, palette=colors_h2, ax=axes[0])
axes[0].set_title('Box-Plot: Çeşitlilik Bazlı Beğeni Dağılımı', fontweight='bold')
axes[0].set_xlabel('Çeşitlilik Grubu')
axes[0].set_ylabel('Beğeni Sayısı (loves_count)')

# Violin Plot
sns.violinplot(x='cesitlilik_grubu', y='loves_count', data=df, palette=colors_h2, ax=axes[1])
axes[1].set_title('Violin Plot: Beğeni Dağılım Yoğunluğu', fontweight='bold')
axes[1].set_xlabel('Çeşitlilik Grubu')
axes[1].set_ylabel('Beğeni Sayısı (loves_count)')

fig.text(0.5, 0.01, f't-Testi: t = {t_stat:.4f}, p(tek yönlü) = {p_val_tek_yonlu:.6f}', ha='center',
         fontsize=12, style='italic',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('grafikler/hipotez2_ttest.png', bbox_inches='tight')
plt.close()
print(f"\n   ✅ Grafik kaydedildi: grafikler/hipotez2_ttest.png")

# ============================================================================
# HİPOTEZ 3: LEVENE TESTİ (VARYANS ANALİZİ)
# "Ekonomik" segmentteki ürünlerin puan varyansı, "Premium"
# segmentteki ürünlerin puan varyansından daha yüksek midir?
# ============================================================================
print("\n" + "-" * 80)
print("HİPOTEZ 3: LEVENE TESTİ (Fiyat Segmentine Göre Puan Varyansı)")
print("-" * 80)

# -------------------------------------------------------------------------
# H₀ : σ²_Ekonomik = σ²_Premium (İki grubun varyansı eşittir)
# H₁ : σ²_Ekonomik > σ²_Premium (Ekonomik segmentte varyans daha yüksek)
# α  : 0.05
#
# Ekonomik Segment : Fiyatı alt %25 diliminde olan ürünler
# Premium Segment  : Fiyatı üst %25 diliminde olan ürünler
#
# Beklenti: Düşük fiyatlı ürünlerde tüketici beklentisi düşük olduğundan
# puanlar daha dağınık (yüksek varyans) olabilir. Premium ürünlerde ise
# yüksek fiyat yüksek kalite beklentisi getirdiğinden puanlar daha tutarlı
# (düşük varyans) olabilir.
# -------------------------------------------------------------------------

# Ekonomik ve Premium segmentleri oluştur (fiyat çeyrekliklerine göre)
ekonomik = df[df['price_usd'] <= fiyat_q1]['rating']
premium = df[df['price_usd'] >= fiyat_q3]['rating']

# Grup istatistikleri
print(f"\n📊 Segment Bazlı Puan İstatistikleri:")
print(f"   Ekonomik Segment (Fiyat ≤ ${fiyat_q1:.2f}):")
print(f"     - Gözlem   : {len(ekonomik)}")
print(f"     - Ortalama : {ekonomik.mean():.4f}")
print(f"     - Varyans  : {ekonomik.var():.4f}")
print(f"     - Std.Sapma: {ekonomik.std():.4f}")
print(f"\n   Premium Segment (Fiyat ≥ ${fiyat_q3:.2f}):")
print(f"     - Gözlem   : {len(premium)}")
print(f"     - Ortalama : {premium.mean():.4f}")
print(f"     - Varyans  : {premium.var():.4f}")
print(f"     - Std.Sapma: {premium.std():.4f}")

# Levene testi uygula
levene_stat, p_val_levene = stats.levene(ekonomik, premium)

print(f"\n📊 Levene Testi Sonuçları:")
print(f"   Levene İstatistiği : {levene_stat:.4f}")
print(f"   p-değeri           : {p_val_levene:.6f}")

if p_val_levene < 0.05:
    print(f"   Karar              : p < 0.05 → H₀ REDDEDİLDİ ❌")
    varyans_karsilastirma = "daha yüksek" if ekonomik.var() > premium.var() else "daha düşük"
    print(f"\n📌 HOCAYA SUNUM NOTU (Hipotez 3 - Levene Testi):")
    print(f"   Levene testi sonucunda W = {levene_stat:.4f}, p = {p_val_levene:.6f} < 0.05")
    print(f"   bulunmuş ve H₀ hipotezi reddedilmiştir.")
    print(f"   Ekonomik segmentteki ürünlerin puan varyansı ({ekonomik.var():.4f}),")
    print(f"   Premium segmenttekinden ({premium.var():.4f}) istatistiksel olarak anlamlı")
    print(f"   derecede {varyans_karsilastirma}tır.")
    if ekonomik.var() > premium.var():
        print(f"\n   Bilimsel Yorum: Bu sonuç, düşük fiyatlı ürünlerde tüketici beklentisinin")
        print(f"   belirsiz ve çeşitli olduğunu göstermektedir. Ekonomik segmentte bazı tüketiciler")
        print(f"   düşük fiyata iyi ürün bulduğunda yüksek puan verirken, kaliteden memnun kalmayanlar")
        print(f"   düşük puan verebilmektedir. Premium segmentte ise yüksek fiyat tutarlı bir kalite")
        print(f"   beklentisi oluşturduğundan puanlar daha homojen dağılmaktadır.")
    else:
        print(f"\n   Bilimsel Yorum: Beklentinin aksine, Premium segmentte puan varyansı daha yüksek")
        print(f"   çıkmıştır. Bu durum, yüksek fiyatın yüksek beklenti oluşturmasından ve")
        print(f"   karşılanmayan beklentilerin düşük puanlara yol açmasından kaynaklanıyor olabilir.")
else:
    print(f"   Karar              : p >= 0.05 → H₀ REDDEDİLEMEDİ ✅")
    print(f"\n📌 HOCAYA SUNUM NOTU (Hipotez 3 - Levene Testi):")
    print(f"   Levene testi sonucunda p = {p_val_levene:.6f} >= 0.05 bulunmuştur.")
    print(f"   H₀ hipotezi reddedilememiş, iki segmentin puan varyansları arasında")
    print(f"   istatistiksel olarak anlamlı bir fark tespit edilememiştir.")

# Hipotez 3 Grafiği
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Hipotez 3: Fiyat Segmentine Göre Puan Değişkenliği (Levene Testi)', fontsize=14, fontweight='bold')

# Violin Plot (Dağılım yoğunluğu)
colors_h3 = ['#3498db', '#e74c3c']
veri_h3 = df[df['price_usd'].apply(lambda x: x <= fiyat_q1 or x >= fiyat_q3)].copy()
veri_h3['segment'] = veri_h3['price_usd'].apply(lambda x: 'Ekonomik' if x <= fiyat_q1 else 'Premium')

sns.violinplot(x='segment', y='rating', data=veri_h3, order=['Ekonomik', 'Premium'],
               palette=colors_h3, ax=axes[0])
axes[0].set_title('Violin Plot: Puan Dağılım Yoğunluğu', fontweight='bold')
axes[0].set_xlabel('Fiyat Segmenti')
axes[0].set_ylabel('Puan (rating)')

# Box-Plot
sns.boxplot(x='segment', y='rating', data=veri_h3, order=['Ekonomik', 'Premium'],
            palette=colors_h3, ax=axes[1])
axes[1].set_title('Box-Plot: Puan Dağılımı', fontweight='bold')
axes[1].set_xlabel('Fiyat Segmenti')
axes[1].set_ylabel('Puan (rating)')

# Varyans bilgisini grafiğe ekle
fig.text(0.5, 0.01,
         f'Levene: W = {levene_stat:.4f}, p = {p_val_levene:.6f} | '
         f'Varyans → Ekonomik: {ekonomik.var():.4f}, Premium: {premium.var():.4f}',
         ha='center', fontsize=11, style='italic',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('grafikler/hipotez3_levene.png', bbox_inches='tight')
plt.close()
print(f"\n   ✅ Grafik kaydedildi: grafikler/hipotez3_levene.png")

# ============================================================================
# HİPOTEZ 4: BAĞIMSIZ ÖRNEKLEM T-TESTİ (Clean at Sephora Etkisi)
# 'Clean at Sephora' etiketine sahip ürünlerin loves_count ortalaması,
# bu etikete sahip olmayan ürünlerden anlamlı derecede yüksek midir?
# ============================================================================
print("\n" + "-" * 80)
print("HİPOTEZ 4: BAĞIMSIZ ÖRNEKLEM T-TESTİ (Temiz İçerik Etkisi)")
print("-" * 80)

# -------------------------------------------------------------------------
# H₀ : μ_Clean = μ_NonClean (Clean at Sephora etkisi yoktur)
# H₁ : μ_Clean > μ_NonClean (Clean ürünler daha çok beğenilir)
# α  : 0.05
# Test Türü: Tek yönlü bağımsız örneklem t-testi
#
# 'Clean at Sephora' etiketi, ürünün zararlı kimyasallar içermediğini
# ve Sephora'nın temiz güzellik standartlarına uygunluğunu belirtir.
# Bu hipotez, tüketicilerin içerik farkındalığının ürün popülaritesine
# etkisini incelemektedir.
# -------------------------------------------------------------------------

# 'highlights' sütununda 'Clean at Sephora' içerenleri tespit et
df['clean_label'] = df['highlights'].fillna('').str.contains('Clean at Sephora', case=False, na=False)
df['clean_grubu'] = df['clean_label'].apply(lambda x: 'Clean at Sephora' if x else 'Diğer')

# Grup istatistikleri
print(f"\n📊 Grup Bazlı Özet İstatistikler:")
print("-" * 60)
clean_ozet = df.groupby('clean_grubu')['loves_count'].agg(['count', 'mean', 'median', 'std']).round(2)
clean_ozet.columns = ['Gözlem Sayısı', 'Ortalama', 'Medyan', 'Std. Sapma']
print(clean_ozet.to_string())

# Grupları ayır
clean_loves = df[df['clean_grubu'] == 'Clean at Sephora']['loves_count']
diger_loves = df[df['clean_grubu'] == 'Diğer']['loves_count']

# Bağımsız örneklem t-testi (Welch's t-test, eşit varyans varsayılmaz)
t_stat_clean, p_val_clean = stats.ttest_ind(clean_loves, diger_loves, equal_var=False)
# Tek yönlü test (H₁: μ_Clean > μ_Diğer)
p_val_clean_tek = p_val_clean / 2

print(f"\n📊 t-Testi Sonuçları:")
print(f"   t-istatistiği : {t_stat_clean:.4f}")
print(f"   p-değeri (çift yönlü) : {p_val_clean:.6f}")
print(f"   p-değeri (tek yönlü)  : {p_val_clean_tek:.6f}")

if p_val_clean_tek < 0.05 and t_stat_clean > 0:
    print(f"   Karar         : p < 0.05 ve t > 0 → H₀ REDDEDİLDİ ❌")
    print(f"\n📌 HOCAYA SUNUM NOTU (Hipotez 4 - Clean at Sephora t-Testi):")
    print(f"   Bağımsız örneklem t-testi sonucunda t = {t_stat_clean:.4f},")
    print(f"   p(tek yönlü) = {p_val_clean_tek:.6f} < 0.05 bulunmuştur.")
    print(f"   H₀ hipotezi reddedilmiştir. 'Clean at Sephora' etiketine sahip ürünlerin")
    print(f"   ortalama beğeni sayısı ({clean_loves.mean():,.0f}), bu etikete sahip olmayan")
    print(f"   ürünlerden ({diger_loves.mean():,.0f}) istatistiksel olarak anlamlı derecede yüksektir.")
    print(f"\n   Tüketici İçerik Farkındalığı Yorumu:")
    print(f"   Bu bulgu, günümüz tüketicilerinin ürün içeriklerine karşı artan farkındalığını")
    print(f"   yansıtmaktadır. 'Clean Beauty' akımı, zararlı kimyasallardan arındırılmış ürünleri")
    print(f"   tercih eden bilinçli tüketicilerin sayısının artmasıyla popülariteyi doğrudan")
    print(f"   etkilemektedir. Markalar için bu durum, temiz içerik sertifikalarının")
    print(f"   rekabet avantajı sağladığını göstermektedir.")
else:
    print(f"   Karar         : H₀ REDDEDİLEMEDİ ✅")
    print(f"\n📌 HOCAYA SUNUM NOTU (Hipotez 4 - Clean at Sephora t-Testi):")
    print(f"   t-testi sonucunda p(tek yönlü) = {p_val_clean_tek:.6f} >= 0.05 bulunmuştur.")
    print(f"   H₀ hipotezi reddedilememiştir. 'Clean at Sephora' etiketinin beğeni sayısı")
    print(f"   üzerinde istatistiksel olarak anlamlı bir etkisi tespit edilememiştir.")
    print(f"   Bu, tüketicilerin satın alma kararlarında içerik farkındalığının henüz")
    print(f"   belirleyici bir faktör olmayabileceğine işaret etmektedir.")

# Hipotez 4 Grafiği
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Hipotez 4: Clean at Sephora Etiketinin Beğeni Üzerindeki Etkisi (t-Testi)',
             fontsize=14, fontweight='bold')

colors_h4 = ['#27ae60', '#95a5a6']

# Box-Plot
sns.boxplot(x='clean_grubu', y='loves_count', data=df,
            order=['Clean at Sephora', 'Diğer'],
            palette=colors_h4, ax=axes[0])
axes[0].set_title('Box-Plot: Clean Etiket Bazlı Beğeni Dağılımı', fontweight='bold')
axes[0].set_xlabel('Etiket Grubu')
axes[0].set_ylabel('Beğeni Sayısı (loves_count)')

# Bar-Plot (Ortalamalar)
grup_ort = df.groupby('clean_grubu')['loves_count'].mean().reindex(['Clean at Sephora', 'Diğer'])
bars = axes[1].bar(['Clean at Sephora', 'Diğer'], grup_ort, color=colors_h4, edgecolor='black', alpha=0.8)
axes[1].set_title('Ortalama Beğeni: Clean vs Diğer', fontweight='bold')
axes[1].set_xlabel('Etiket Grubu')
axes[1].set_ylabel('Ortalama Beğeni (loves_count)')
for bar_item, val in zip(bars, grup_ort):
    axes[1].text(bar_item.get_x() + bar_item.get_width()/2., bar_item.get_height() + 100,
                f'{val:,.0f}', ha='center', va='bottom', fontweight='bold')

fig.text(0.5, 0.01, f't-Testi: t = {t_stat_clean:.4f}, p(tek yönlü) = {p_val_clean_tek:.6f}',
         ha='center', fontsize=12, style='italic',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('grafikler/hipotez4_clean.png', bbox_inches='tight')
plt.close()
print(f"\n   ✅ Grafik kaydedildi: grafikler/hipotez4_clean.png")

# ============================================================================
# BÖLÜM 7: KORELASYON MATRİSİ (Ek Analiz)
# ============================================================================
print("\n" + "-" * 80)
print("EK ANALİZ: KORELASYON MATRİSİ")
print("-" * 80)

# Sayısal değişkenler arasındaki ilişkiyi gösteren Pearson korelasyonu
korelasyon_degiskenleri = ['loves_count', 'rating', 'price_usd', 'reviews', 'child_count']
korelasyon_matrisi = df[korelasyon_degiskenleri].corr().round(4)

print(f"\n📊 Pearson Korelasyon Matrisi:")
print(korelasyon_matrisi.to_string())

# Korelasyon Isı Haritası
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(korelasyon_matrisi, annot=True, cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, fmt='.3f',
            cbar_kws={'shrink': 0.8}, ax=ax)
ax.set_title('Sayısal Değişkenler Arası Korelasyon Matrisi', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('grafikler/korelasyon_matrisi.png', bbox_inches='tight')
plt.close()
print(f"\n   ✅ Grafik kaydedildi: grafikler/korelasyon_matrisi.png")

# ============================================================================
# BÖLÜM 8: KATEGORİ BAZLI ANALİZ (Ek Grafik)
# ============================================================================
print("\n" + "-" * 80)
print("EK ANALİZ: KATEGORİ BAZLI DAĞILIM")
print("-" * 80)

# Ana kategorilere göre ürün sayısı ve ortalama beğeni
kategori_analiz = df.groupby('primary_category').agg(
    urun_sayisi=('product_id', 'count'),
    ort_begeni=('loves_count', 'mean'),
    ort_fiyat=('price_usd', 'mean')
).round(2).sort_values('urun_sayisi', ascending=False)

print(f"\n📊 Ana Kategori Bazlı Özet:")
print(kategori_analiz.to_string())

# Kategori bazlı bar grafiği
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Ana Kategoriye Göre Ürün Dağılımı', fontsize=14, fontweight='bold')

# Ürün Sayısı
top_kategoriler = kategori_analiz.head(8)
bars1 = axes[0].barh(top_kategoriler.index, top_kategoriler['urun_sayisi'],
                     color=sns.color_palette('viridis', len(top_kategoriler)), edgecolor='black')
axes[0].set_title('Ürün Sayısı (İlk 8 Kategori)', fontweight='bold')
axes[0].set_xlabel('Ürün Sayısı')
axes[0].invert_yaxis()

# Ortalama Beğeni
bars2 = axes[1].barh(top_kategoriler.index, top_kategoriler['ort_begeni'],
                     color=sns.color_palette('magma', len(top_kategoriler)), edgecolor='black')
axes[1].set_title('Ortalama Beğeni (İlk 8 Kategori)', fontweight='bold')
axes[1].set_xlabel('Ortalama Beğeni Sayısı')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('grafikler/kategori_analiz.png', bbox_inches='tight')
plt.close()
print(f"\n   ✅ Grafik kaydedildi: grafikler/kategori_analiz.png")

# ============================================================================
# SONUÇ ÖZETİ
# ============================================================================
print("\n" + "=" * 80)
print("GENEL SONUÇ ÖZETİ")
print("=" * 80)

print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│ VERİ SETİ BİLGİLERİ                                                    │
├─────────────────────────────────────────────────────────────────────────┤
│ Orijinal veri boyutu  : {df_raw.shape[0]:>6} satır × {df_raw.shape[1]:>2} sütun                       │
│ Temizlik sonrası boyut: {len(df):>6} satır × {df.shape[1]:>2} sütun                       │
│ Eksik veri doldurma   : Sayısal→Median, Kategorik→Mode                 │
│ Aykırı değer yöntemi  : IQR (1.5×IQR kuralı)                          │
├─────────────────────────────────────────────────────────────────────────┤
│ HİPOTEZ TEST SONUÇLARI                                                 │
├─────────────────────────────────────────────────────────────────────────┤
│ H1 (ANOVA)  : F={f_stat:>8.4f}, p={p_val_anova:>10.6f}  →  {'H₀ Reddedildi ❌' if p_val_anova < 0.05 else 'H₀ Reddedilemedi ✅':>22}  │
│ H2 (t-Test) : t={t_stat:>8.4f}, p={p_val_tek_yonlu:>10.6f}  →  {'H₀ Reddedildi ❌' if p_val_tek_yonlu < 0.05 and t_stat > 0 else 'H₀ Reddedilemedi ✅':>22}  │
│ H3 (Levene) : W={levene_stat:>8.4f}, p={p_val_levene:>10.6f}  →  {'H₀ Reddedildi ❌' if p_val_levene < 0.05 else 'H₀ Reddedilemedi ✅':>22}  │
│ H4 (t-Test) : t={t_stat_clean:>8.4f}, p={p_val_clean_tek:>10.6f}  →  {'H₀ Reddedildi ❌' if p_val_clean_tek < 0.05 and t_stat_clean > 0 else 'H₀ Reddedilemedi ✅':>22}  │
├─────────────────────────────────────────────────────────────────────────┤
│ Tüm grafikler 'grafikler/' klasörüne kaydedilmiştir.                   │
└─────────────────────────────────────────────────────────────────────────┘
""")

print("=" * 80)
print("ANALİZ TAMAMLANDI")
print("=" * 80)
