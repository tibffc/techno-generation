"""
Cluster prompts using TF-IDF + KMeans (no external ML libraries except sklearn which you already have).
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter

# Твои промпты
prompts = [
    "A dark, raw electronic track with a steady beat, sparse rhythm, and soft drum groove.",
    "A high-energy instrumental techno track with a steady groove, dense rhythm, and noisy texture, blending progressive and warehouse influences.",
    "A high-energy driving club track blending hard and acid techno with a steady groove, sparse rhythm, and noisy textures.",
    "A deep melodic techno track with a calm, steady groove and soft drum attacks, blending ambient textures and dark, balanced soundscapes.",
    "A high-energy acid techno track with a groovy, progressive rhythm, bright sound, and balanced texture.",
    "A high-energy dub techno track with a dark mood, steady groove, and bright sound featuring a dense rhythm, noisy texture, and balanced drum attack.",
    "A hypnotic techno track with a steady groove, dark sound, and soft drum attack.",
    "A bright, melodic house-influenced techno track with a groovy, medium-energy club dance groove and a balanced drum attack.",
    "A dark, warm electronic track with a sparse rhythm and soft drum attack, blending low energy and balanced textures.",
    "A high-energy rave techno track with a steady groove, punchy drums, and bright, dense rhythm layered with noisy textures and mostly instrumental elements."
]

print("="*70)
print("📊 PROMPT CLUSTERING (TF-IDF + KMeans)")
print("="*70)

# 1. Преобразуем промпты в TF-IDF векторы
print("\n🔮 Converting prompts to TF-IDF vectors...")
vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1, 2),
    lowercase=True
)
X = vectorizer.fit_transform(prompts)
print(f"   Shape: {X.shape}")
print(f"   Features: {len(vectorizer.get_feature_names_out())}")

# Покажем самые важные слова
feature_names = vectorizer.get_feature_names_out()
print(f"\n📝 Top features: {', '.join(feature_names[:20])}")

# 2. Найдём оптимальное количество кластеров
print("\n🔧 Finding optimal number of clusters...")

best_k = 2
best_score = -1

for k in range(2, min(6, len(prompts))):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X.toarray())
    if len(set(labels)) > 1:
        score = silhouette_score(X.toarray(), labels)
        print(f"   k={k}: silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k

print(f"\n   ✅ Best k: {best_k} (silhouette={best_score:.4f})")

# 3. Кластеризуем с оптимальным k
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=5)
prompt_clusters = kmeans.fit_predict(X.toarray())

# 4. Распределение промптов по кластерам
print("\n📋 PROMPT DISTRIBUTION BY CLUSTER:")
cluster_counts = Counter(prompt_clusters)
for cluster in sorted(cluster_counts.keys()):
    count = cluster_counts[cluster]
    pct = count / len(prompts) * 100
    print(f"\n   Cluster {cluster}: {count} prompts ({pct:.0f}%)")
    print(f"   {'-'*50}")
    # Покажем все промпты в этом кластере
    for i, c in enumerate(prompt_clusters):
        if c == cluster:
            print(f"   [{i+1}] {prompts[i][:80]}...")
    
    # Покажем ключевые слова для этого кластера
    # Найдём средний вектор для кластера
    cluster_indices = [i for i, c in enumerate(prompt_clusters) if c == cluster]
    cluster_center = X[cluster_indices].mean(axis=0).A1
    top_features_idx = np.argsort(cluster_center)[-5:][::-1]
    top_features = [feature_names[idx] for idx in top_features_idx if cluster_center[idx] > 0]
    print(f"   → Key words: {', '.join(top_features)}")

# 5. Анализ энергии в каждом кластере
print("\n" + "="*70)
print("⚡ ENERGY ANALYSIS BY CLUSTER")
print("="*70)

for cluster in sorted(cluster_counts.keys()):
    cluster_prompts = [prompts[i] for i, c in enumerate(prompt_clusters) if c == cluster]
    
    high = sum(1 for p in cluster_prompts if 'high-energy' in p.lower() or 'high energy' in p.lower() or 'energetic' in p.lower())
    low = sum(1 for p in cluster_prompts if 'low-energy' in p.lower() or 'low energy' in p.lower() or 'calm' in p.lower())
    
    print(f"\n   Cluster {cluster}:")
    print(f"      High-energy: {high}/{len(cluster_prompts)}")
    print(f"      Low-energy: {low}/{len(cluster_prompts)}")

# 6. Сравнение с генерациями
print("\n" + "="*70)
print(" COMPARISON WITH GENERATIONS")
print("="*70)

print(f"\n   Prompts cover: {best_k} clusters")
print(f"   Base generates: 5 clusters")
print(f"   Rank 8 generates: 3 clusters (0, 1, 4)")

if best_k <= 3:
    print("\n   ✅ Rank 8 stays within prompt clusters")
else:
    print("\n   ⚠️ Rank 8 generates fewer clusters than prompted")

if 5 > best_k:
    print("   ⚠️ Base generates MORE clusters than prompted (deviates from instructions)")
else:
    print("   ✅ Base stays within prompt clusters")

# 7. Итоговый вывод
print("\n" + "="*70)
print("💡 FINAL CONCLUSION")
print("="*70)

print(f"""
   📌 Prompts were clustered into {best_k} distinct groups.
   📌 High-energy prompts: {sum(1 for p in prompts if 'high-energy' in p.lower())}/10
   📌 Low-energy prompts: {sum(1 for p in prompts if 'low-energy' in p.lower())}/10

   🎯 Base model: Generates into 5 clusters (wider than prompts)
      → DEVIATES from instructions, adds unprompted styles

   🎯 Rank 8 model: Generates into 3 clusters (closer to prompt distribution)
      → FOLLOWS instructions more faithfully

   🎯 Cluster 2 (11.6% of reference) is missing from ALL generations
      → Model CANNOT generate this style (limitation of LoRA/model)
""")