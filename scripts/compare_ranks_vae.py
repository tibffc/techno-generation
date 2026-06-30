"""
Compare LoRA ranks (base, rank4_2ep, rank8, rank16) using Spotify VAE.

Usage:
    python scripts/compare_ranks_vae_spotify.py
"""

import sys
import json
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.diversity.extractor_universal import AudioFeatureExtractor
from scripts.diversity.analyzer import DiversityAnalyzer
from scripts.diversity.vae_trainer import VAE


# ============================================================
# CONFIGURATION
# ============================================================
RANKS = ['base_params', 'rank4_2ep_params', 'rank8_params']
RANK_NAMES = {
    'base_params': 'Base (no LoRA)',
    'rank4_2ep_params': 'LoRA Rank 4 (2 epochs)',
    'rank8_params': 'LoRA Rank 8',
}

# Paths
REFERENCE_EMBEDDINGS_PATH = Path('data/embeddings_spotify.npy')
VAE_MODEL_PATH = Path('outputs/vae_model_spotify/best_model.pt')
GENERATED_BASE_DIR = Path('data/organized_by_rank2')
OUTPUT_DIR = Path('outputs/ranks_comparison_spotify_vae')


def compress_embeddings(embeddings_768, model, device, batch_size=128):
    """Compress 768-dim embeddings to latent space using VAE."""
    compressed = []
    with torch.no_grad():
        for i in range(0, len(embeddings_768), batch_size):
            batch = torch.FloatTensor(embeddings_768[i:i+batch_size]).to(device)
            mu, _ = model.encode(batch)
            compressed.append(mu.cpu().numpy())
    return np.concatenate(compressed, axis=0)


def load_or_extract_embeddings(generated_dir, extractor, rank_name):
    """Extract embeddings from generated files."""
    print(f"\n   Extracting embeddings for {rank_name}...")
    embeddings_768, paths = extractor.extract_from_directory(generated_dir)
    print(f"      Extracted {len(embeddings_768)} embeddings, dim=768")
    return embeddings_768, paths


def plot_clusters_2d(embeddings, labels, rank_name, output_path, metrics):
    """Plot 2D t-SNE visualization of clusters."""
    print(f"\n   Visualizing {rank_name}...")
    
    n_samples = len(embeddings)
    perplexity = min(30, n_samples - 1) if n_samples > 1 else 1
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Добавляем информацию о креативных метриках в заголовок
    creativity = metrics.get('creativity', {})
    title = f'{RANK_NAMES[rank_name]}\n'
    title += f'Coverage: {metrics["coverage_percentage"]:.1f}% | '
    title += f'Diversity: {metrics["diversity_score"]:.3f} | '
    title += f'Clusters: {metrics["optimal_clusters"]}\n'
    if creativity:
        title += f'CTR: {creativity.get("ctr", 0):.2f} | '
        title += f'S-score: {creativity.get("s_score", 0):.2f} | '
        title += f'NCI: {creativity.get("nci", 0):.2f}'
    
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels, cmap='tab10', s=30, alpha=0.6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    cbar = plt.colorbar(scatter, ax=ax, label='Cluster ID')
    cbar.set_ticks(np.unique(labels))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      Plot saved to {output_path}")


def plot_creativity_comparison(df, output_path):
    """Plot creativity metrics comparison bar chart."""
    if df.empty or 'creativity_ctr' not in df.columns:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # CTR - гибридность
    axes[0, 0].bar(df['rank_name'], df['creativity_ctr'], color=['gray', 'blue', 'orange'])
    axes[0, 0].set_ylabel('CTR (Cluster Transition Rate)')
    axes[0, 0].set_title('Hybridity (higher = more hybrids)')
    axes[0, 0].axhline(y=1.5, color='green', linestyle='--', label='Hybrid threshold')
    axes[0, 0].legend()
    
    # S-score - неожиданность
    axes[0, 1].bar(df['rank_name'], df['creativity_s_score'], color=['gray', 'blue', 'orange'])
    axes[0, 1].set_ylabel('S-score (Surprise)')
    axes[0, 1].set_title('Unexpectedness (higher = more surprising)')
    
    # NCI - редкость комбинаций
    axes[1, 0].bar(df['rank_name'], df['creativity_nci'], color=['gray', 'blue', 'orange'])
    axes[1, 0].set_ylabel('NCI (Novel Combination Index)')
    axes[1, 0].set_title('Rarity of combinations (higher = rarer)')
    
    # CLD - резкость переходов
    axes[1, 1].bar(df['rank_name'], df['creativity_cld'], color=['gray', 'blue', 'orange'])
    axes[1, 1].set_ylabel('CLD (Cluster Leap Distance)')
    axes[1, 1].set_title('Transition sharpness (higher = sharper changes)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Creativity comparison chart saved to {output_path}")


def plot_comparison_table(df, output_path):
    """Create a visual table of metrics comparison."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Основные метрики
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            row['rank_name'],
            f"{row['optimal_clusters']}",
            f"{row['coverage_percentage']:.1f}%",
            f"{row['diversity_score']:.3f}",
            f"{row['entropy_ratio']:.3f}",
            f"{row['perplexity']:.2f}",
            f"{row['novelty_percentage']:.1f}%",
            f"{row.get('creativity_ctr', 0):.2f}",
            f"{row.get('creativity_s_score', 0):.2f}",
            f"{row.get('creativity_nci', 0):.2f}",
        ])
    
    columns = ['Rank', 'Clusters', 'Coverage', 'Diversity', 'Entropy', 'Perplexity', 'Novelty', 'CTR', 'S-score', 'NCI']
    table = ax.table(cellText=table_data, colLabels=columns, loc='center',
                     cellLoc='center', colColours=['#4472C4']*len(columns))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    ax.set_title('LoRA Ranks Comparison (Spotify VAE 32d) with Creativity Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   Comparison table saved to {output_path}")


def plot_metrics_radar(df, output_path):
    """Plot radar chart comparing all ranks."""
    categories = ['Coverage', 'Diversity', 'Entropy Ratio', 'Silhouette', 'CTR', 'S-score', 'NCI']
    
    # Нормализуем значения для радара
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    
    colors = {'base_params': 'gray', 'rank4_2ep_params': 'blue', 'rank8_params': 'orange'}
    
    for _, row in df.iterrows():
        values = [
            row['coverage_percentage'] / 100,
            row['diversity_score'],
            row['entropy_ratio'],
            row['silhouette_score'],
            min(row.get('creativity_ctr', 0) / 2.5, 1.0),
            min(row.get('creativity_s_score', 0) / 2.0, 1.0),
            min(row.get('creativity_nci', 0) / 0.5, 1.0),
        ]
        values += values[:1]
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, 
                label=row['rank_name'], color=colors[row['rank']])
        ax.fill(angles, values, alpha=0.15, color=colors[row['rank']])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('LoRA Ranks Comparison Radar Chart (with Creativity Metrics)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Radar chart saved to {output_path}")


def plot_bar_comparison(df, output_path):
    """Plot bar chart comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Coverage
    axes[0].bar(df['rank_name'], df['coverage_percentage'], color=['gray', 'blue', 'orange'])
    axes[0].set_ylabel('Coverage (%)')
    axes[0].set_title('Coverage by Rank')
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(df['coverage_percentage']):
        axes[0].text(i, v + 2, f'{v:.1f}%', ha='center')
    
    # Diversity Score
    axes[1].bar(df['rank_name'], df['diversity_score'], color=['gray', 'blue', 'orange'])
    axes[1].set_ylabel('Diversity Score')
    axes[1].set_title('Diversity Score by Rank')
    axes[1].set_ylim(0, 1)
    for i, v in enumerate(df['diversity_score']):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # Entropy Ratio
    axes[2].bar(df['rank_name'], df['entropy_ratio'], color=['gray', 'blue', 'orange'])
    axes[2].set_ylabel('Entropy Ratio')
    axes[2].set_title('Entropy Ratio by Rank')
    axes[2].set_ylim(0, 1)
    axes[2].axhline(y=1.0, color='green', linestyle='--', label='Ideal (1.0)')
    axes[2].legend()
    for i, v in enumerate(df['entropy_ratio']):
        axes[2].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Bar chart saved to {output_path}")


def main():
    print("\n" + "="*70)
    print(" LoRA RANKS COMPARISON WITH SPOTIFY VAE 32d")
    print("="*70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n Device: {device}")
    
    # Load VAE model
    print(f"\n Loading VAE model from {VAE_MODEL_PATH}...")
    checkpoint = torch.load(VAE_MODEL_PATH, map_location=device)
    latent_dim = checkpoint['latent_dim']
    model = VAE(input_dim=768, latent_dim=latent_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"   VAE loaded (latent_dim={latent_dim})")
    
    # Load reference embeddings
    print(f"\n Loading reference embeddings from {REFERENCE_EMBEDDINGS_PATH}...")
    ref_embeddings_768 = np.load(REFERENCE_EMBEDDINGS_PATH)
    print(f"   Loaded {len(ref_embeddings_768)} embeddings, dim=768")
    
    # Compress reference embeddings
    print(f"\n Compressing reference embeddings...")
    ref_embeddings = compress_embeddings(ref_embeddings_768, model, device)
    print(f"   Compressed to {ref_embeddings.shape}")
    
    # Fit reference once for all ranks
    print(f"\n Fitting reference clusterer...")
    analyzer = DiversityAnalyzer(
        min_clusters=2, 
        max_clusters=20,
        cache_path='cache/reference_spotify.pkl'
    )
    analyzer.fit_reference(ref_embeddings)
    print(f"   Optimal clusters: {analyzer.n_clusters_optimal}")
    print(f"   Silhouette score: {analyzer.silhouette:.4f}")
    
    # Extractor for generated files
    extractor = AudioFeatureExtractor(backend='wav2vec2_music', duration=10.0)
    
    results = []
    all_embeddings = {}
    all_labels = {}
    all_metrics = {}
    
    for rank in RANKS:
        print("\n" + "-"*50)
        print(f" Processing: {RANK_NAMES[rank]}")
        print("-"*50)
        
        generated_dir = GENERATED_BASE_DIR / rank
        
        if not generated_dir.exists():
            print(f"    Directory not found: {generated_dir}")
            continue
        
        gen_embeddings_768, _ = load_or_extract_embeddings(generated_dir, extractor, rank)
        gen_embeddings = compress_embeddings(gen_embeddings_768, model, device)
        
        metrics = analyzer.validate_generation(gen_embeddings)
        metrics['rank'] = rank
        metrics['rank_name'] = RANK_NAMES[rank]
        
        # Извлекаем креативные метрики
        creativity = metrics.get('creativity', {})
        
        all_metrics[rank] = metrics
        all_embeddings[rank] = gen_embeddings
        all_labels[rank] = analyzer.kmeans.predict(analyzer.scaler.transform(gen_embeddings))
        
        results.append({
            'rank': rank,
            'rank_name': RANK_NAMES[rank],
            'optimal_clusters': metrics['optimal_clusters'],
            'silhouette_score': metrics['silhouette_score'],
            'coverage_percentage': metrics['coverage_percentage'],
            'diversity_score': metrics['diversity_score'],
            'entropy_ratio': metrics['entropy_ratio'],
            'perplexity': metrics['perplexity'],
            'novelty_percentage': metrics['novelty_percentage'],
            'n_generated_samples': metrics['n_generated_samples'],
            # Креативные метрики
            'creativity_ctr': creativity.get('ctr', 0),
            'creativity_hybrid_percent': creativity.get('hybrid_percent', 0),
            'creativity_s_score': creativity.get('s_score', 0),
            'creativity_high_surprise_percent': creativity.get('high_surprise_percent', 0),
            'creativity_nci': creativity.get('nci', 0),
            'creativity_rare_combinations_percent': creativity.get('rare_combinations_percent', 0),
            'creativity_cld': creativity.get('cld', 0),
            'creativity_transition_rate': creativity.get('transition_rate', 0),
        })
        
        print(f"\n    RESULTS for {rank}:")
        print(f"      Coverage: {metrics['coverage_percentage']:.1f}%")
        print(f"      Diversity score: {metrics['diversity_score']:.3f}")
        print(f"      Entropy ratio: {metrics['entropy_ratio']:.3f}")
        print(f"      Perplexity: {metrics['perplexity']:.2f}")
        if creativity:
            print(f"\n    CREATIVITY METRICS:")
            creativity = metrics.get('creativity', {})
            print(f"      CTR (hybridity): {creativity.get('ctr', 0):.2f}")
            print(f"      Hybrid %: {creativity.get('hybrid_percent', 0):.1f}%")
            print(f"      Extreme hybrid %: {creativity.get('extreme_hybrid_percent', 0):.1f}%")
            print(f"      S-score (surprise): {creativity.get('s_score', 0):.2f}")
            print(f"      High surprise %: {creativity.get('high_surprise_percent', 0):.1f}%")
            print(f"      Intra-cluster richness: {creativity.get('intra_cluster_richness', 0):.4f}")
            print(f"      Cross-cluster novelty: {creativity.get('cross_cluster_novelty', 0):.1f}%")
            print(f"      Creativity Index: {creativity.get('creativity_index', 0):.3f}")
            print(f"      Surprise/Coverage Ratio: {creativity.get('surprise_to_coverage_ratio', 0):.2f}")
        
        report_path = OUTPUT_DIR / f'{rank}_report.json'
        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"      Report saved to {report_path}")
        
        plot_path = OUTPUT_DIR / f'{rank}_clusters.png'
        plot_clusters_2d(gen_embeddings, all_labels[rank], rank, plot_path, metrics)
    
    print(f"\n📊 DEBUG: Collected {len(results)} results")
    for r in results:
        print(f"   - {r['rank_name']}: {r['n_generated_samples']} samples")
    
    if len(results) == 0:
        print("❌ ERROR: No results! Check if directories exist and contain audio files.")
        return
    
    try:
        df = pd.DataFrame(results)
        print(f"   DataFrame shape: {df.shape}")
        
        # Save CSV and JSON
        csv_path = OUTPUT_DIR / 'ranks_comparison.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n📊 CSV saved to {csv_path}")
        
        json_path = OUTPUT_DIR / 'ranks_comparison.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📊 JSON saved to {json_path}")
        
        # Print final table
        print("\n" + "="*110)
        print("📊 FINAL COMPARISON TABLE")
        print("="*110)
        print(f"{'Rank':<25} {'Clusters':<8} {'Coverage':<9} {'Diversity':<9} {'Entropy':<8} {'Perplexity':<9} {'Novelty':<8} {'CTR':<7} {'S-score':<8} {'NCI':<7}")
        print("-"*110)
        for _, row in df.iterrows():
            print(f"{row['rank_name']:<25} {row['optimal_clusters']:<8} {row['coverage_percentage']:<9.1f}% "
                  f"{row['diversity_score']:<9.3f} {row['entropy_ratio']:<8.3f} {row['perplexity']:<9.2f} "
                  f"{row['novelty_percentage']:<8.1f}% {row['creativity_ctr']:<7.2f} {row['creativity_s_score']:<8.2f} {row['creativity_nci']:<7.4f}")
        print("="*110)
        
        # Visualizations
        print("\n🎨 Creating visualizations...")
        plot_comparison_table(df, OUTPUT_DIR / 'comparison_table.png')
        plot_metrics_radar(df, OUTPUT_DIR / 'radar_chart.png')
        plot_bar_comparison(df, OUTPUT_DIR / 'bar_chart.png')
        plot_creativity_comparison(df, OUTPUT_DIR / 'creativity_chart.png')
        
        # Recommendations
        if len(df) > 0:
            best_diversity = df.loc[df['diversity_score'].idxmax()]
            best_coverage = df.loc[df['coverage_percentage'].idxmax()]
            best_creativity = df.loc[df['creativity_ctr'].idxmax()]
            
            print("\n🏆 RECOMMENDATIONS:")
            print(f"   Best diversity score: {best_diversity['rank_name']} ({best_diversity['diversity_score']:.3f})")
            print(f"   Best coverage: {best_coverage['rank_name']} ({best_coverage['coverage_percentage']:.1f}%)")
            print(f"   Most creative (highest CTR): {best_creativity['rank_name']} ({best_creativity['creativity_ctr']:.2f})")
            
            if best_diversity['rank'] == best_coverage['rank'] == best_creativity['rank']:
                print(f"   🎯 RECOMMENDED RANK: {best_diversity['rank_name']}")
            else:
                print(f"   ⚠️ Trade-off: best diversity ({best_diversity['rank_name']}), "
                      f"best coverage ({best_coverage['rank_name']}), "
                      f"most creative ({best_creativity['rank_name']})")
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    return results


if __name__ == "__main__":
    main()