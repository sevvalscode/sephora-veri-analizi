import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols

# DOSYA YOLU AYARI: Python'ın dosyayı bulmasını garanti eder
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
except:
    pass



try:
    df =  pd.read_excel('product_info.xlsx')
    print("Veri başarıyla yüklendi!")
except Exception as e:
    print(f"Hata: Dosya yüklenemedi. Sebebi: {e}")

# Veri Temizliği
df = df.dropna(subset=['price_usd', 'loves_count', 'rating', 'brand_name', 'child_count'])
df['brand_name'] = df['brand_name'].astype(str).str.strip()

# --- HİPOTEZ 1: ANOVA (Marka Segmenti) ---
def segment_brand(brand):
    luxury = ['Chanel', 'Dior', 'Yves Saint Laurent', 'Guerlain']
    niche = ['19-69', 'Byredo', 'Diptyque']
    if any(x in brand for x in luxury): return 'Luxury'
    if any(x in brand for x in niche): return 'Niche'
    return 'Mass Market'

df['brand_segment'] = df['brand_name'].apply(segment_brand)

model = ols('loves_count ~ brand_segment', data=df).fit()
print("\n--- ANOVA Sonuçları ---")
print(sm.stats.anova_lm(model, typ=2))

plt.figure(figsize=(10, 5))
sns.boxplot(x='brand_segment', y='loves_count', data=df)
plt.yscale('log')
plt.title('Hipotez 1: Marka Segmentine Göre Beğeni')
plt.show()

# --- HİPOTEZ 2: t-testi (Varyasyon Sayısı) ---
df['variation_group'] = df['child_count'].apply(lambda x: 'Çok Çeşitli (>5)' if x > 5 else 'Tekil')
group1 = df[df['variation_group'] == 'Çok Çeşitli (>5)']['loves_count']
group2 = df[df['variation_group'] == 'Tekil']['loves_count']
t_stat, p_val = stats.ttest_ind(group1, group2)

print(f"\n--- t-Testi Sonuçları ---\nt-istatistiği: {t_stat:.4f}, p-değeri: {p_val:.4f}")

plt.figure(figsize=(8, 5))
sns.barplot(x='variation_group', y='loves_count', data=df)
plt.title('Hipotez 2: Ürün Çeşitliliği Etkisi')
plt.show()
df['price_usd'] = pd.to_numeric(df['price_usd'], errors='coerce')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# --- HİPOTEZ 3: Levene Testi (Puan Varyansı) ---
q1 = df['price_usd'].quantile(0.25)
q3 = df['price_usd'].quantile(0.75)
df['price_segment'] = df['price_usd'].apply(lambda x: 'Ekonomik' if x <= q1 else ('Premium' if x >= q3 else 'Orta'))

eco_ratings = df[df['price_segment'] == 'Ekonomik']['rating'].dropna()
prem_ratings = df[df['price_segment'] == 'Premium']['rating'].dropna()

l_stat, l_p = stats.levene(eco_ratings, prem_ratings)

print(f"\n--- Levene Testi Sonucu ---\np-değeri: {l_p:.4f}")

plt.figure(figsize=(10, 5))
sns.violinplot(x='price_segment', y='rating', data=df[df['price_segment'].isin(['Ekonomik', 'Premium'])])
plt.title('Hipotez 3: Fiyat Segmentine Göre Puan Değişkenliği')
plt.show()

# --- KRİTİK AYAR: Dosyayı bulmanı sağlar ---
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
# ------------------------------------------

try:
    # Dosya adın tam olarak 'veri.xlsx' ise:
    df = pd.read_excel('veri.xlsx')
    print("Veri başarıyla yüklendi!")
    
    # Analizlere devam et...
    # (Daha önce verdiğim analiz kodlarını buraya yapıştırabilirsin)
    
except Exception as e:
    print(f"Hata oluştu: {e}")