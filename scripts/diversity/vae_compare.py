import sys
import argparse
import json
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.diversity.extractor_universal import AudioFeatureExtractor
from scripts.diversity.analyzer import DiversityAnalyzer
from scripts.diversity.vae_trainer import VAE


def plot_latent_space(latent_embeddings, labels, output_path, title="VAE Latent Space", 
                      silhouette_score=None, diversity_score=None):
    """Plot latent space with t-SNE projection."""
    print(f"\n Visualizing latent space with t-SNE...")
    
    n_samples = len(latent_embeddings)
    perplexity = min(30, n_samples - 1) if n_samples > 1 else 1
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    latent_2d = tsne.fit_transform(latent_embeddings)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Colored by cluster
    sc1 = axes[0].scatter(latent_2d[:, 0], latent_2d[:, 1], 
                          c=labels, cmap='tab10', s=8, alpha=0.5)
    axes[0].set_title(f'{title}\nColored by Cluster ID', fontsize=12)
    axes[0].set_xlabel('t-SNE Component 1')
    axes[0].set_ylabel('t-SNE Component 2')
    cbar = plt.colorbar(sc1, ax=axes[0], label='Cluster ID')
    cbar.set_ticks(np.unique(labels))
    
    # Add text with metrics
    if silhouette_score:
        axes[0].text(0.02, 0.98, f'Silhouette: {silhouette_score:.4f}', 
                    transform=axes[0].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    if diversity_score:
        axes[0].text(0.02, 0.92, f'Diversity: {diversity_score:.4f}', 
                    transform=axes[0].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: All points
    axes[1].scatter(latent_2d[:, 0], latent_2d[:, 1], s=20, alpha=0.5, c='steelblue')
    axes[1].set_title(f'{title}\nAll Samples (n={len(latent_embeddings)})', fontsize=12)
    axes[1].set_xlabel('t-SNE Component 1')
    axes[1].set_ylabel('t-SNE Component 2')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Plot saved to: {output_path}")


def plot_comparison_bar(metrics_orig, metrics_latent, output_path):
    """Plot comparison bar chart of all metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define metrics to compare
    metric_names = [
        'Silhouette Score', 
        'Coverage', 
        'Entropy Ratio', 
        'Diversity Score',
        'Intra-cluster Variance (ref)'
    ]
    
    orig_values = [
        metrics_orig.get('silhouette_score', 0),
        metrics_orig.get('coverage', 0),
        metrics_orig.get('entropy_ratio', 0),
        metrics_orig.get('diversity_score', 0),
        metrics_orig.get('intra_cluster_variance_reference', 0)
    ]
    
    latent_values = [
        metrics_latent.get('silhouette_score', 0),
        metrics_latent.get('coverage', 0),
        metrics_latent.get('entropy_ratio', 0),
        metrics_latent.get('diversity_score', 0),
        metrics_latent.get('intra_cluster_variance_reference', 0)
    ]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, orig_values, width, label=f'Original ({metrics_orig.get("n_reference_samples", "?")} samples, 768d)', color='steelblue')
    bars2 = ax.bar(x + width/2, latent_values, width, label=f'VAE-Compressed (32d)', color='coral')
    
    ax.set_ylabel('Score')
    ax.set_title('Clustering Quality: Original vs VAE-Compressed')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=15, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.1)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Comparison plot saved to: {output_path}")


def plot_metrics_radar(metrics_orig, metrics_latent, output_path):
    """Plot radar chart for multi-dimensional comparison."""
    categories = ['Silhouette', 'Coverage', 'Entropy', 'Diversity', 'Quality']
    
    orig_values = [
        metrics_orig.get('silhouette_score', 0),
        metrics_orig.get('coverage', 0),
        metrics_orig.get('entropy_ratio', 0),
        metrics_orig.get('diversity_score', 0),
        1 - metrics_orig.get('intra_cluster_variance_reference', 0)  # invert because lower is better
    ]
    
    latent_values = [
        metrics_latent.get('silhouette_score', 0),
        metrics_latent.get('coverage', 0),
        metrics_latent.get('entropy_ratio', 0),
        metrics_latent.get('diversity_score', 0),
        1 - metrics_latent.get('intra_cluster_variance_reference', 0)
    ]
    
    # Normalize to 0-1 range for radar
    def normalize_radar(vals):
        max_val = max(max(orig_values), max(latent_values)) or 1
        return [v / max_val for v in vals]
    
    orig_norm = normalize_radar(orig_values)
    latent_norm = normalize_radar(latent_values)
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    orig_norm += orig_norm[:1]
    latent_norm += latent_norm[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    ax.plot(angles, orig_norm, 'o-', linewidth=2, label='Original', color='steelblue')
    ax.fill(angles, orig_norm, alpha=0.25, color='steelblue')
    ax.plot(angles, latent_norm, 'o-', linewidth=2, label='VAE', color='coral')
    ax.fill(angles, latent_norm, alpha=0.25, color='coral')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Metrics Comparison Radar Chart')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Radar chart saved to: {output_path}")


def compare_clustering(embeddings, vae_model_path, output_dir='outputs/vae_comparison'):
    """Compare clustering quality before and after VAE compression."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load VAE model
    print("\n Loading VAE model...")
    checkpoint = torch.load(vae_model_path, map_location=device)
    latent_dim = checkpoint['latent_dim']
    model = VAE(input_dim=embeddings.shape[1], latent_dim=latent_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Compress to latent space
    print(f" Compressing to latent space (dim={latent_dim})...")
    latent_embeddings = []
    batch_size = 128
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = torch.FloatTensor(embeddings[i:i+batch_size]).to(device)
            mu, _ = model.encode(batch)
            latent_embeddings.append(mu.cpu().numpy())
    latent_embeddings = np.concatenate(latent_embeddings, axis=0)
    print(f"   Compression complete: {embeddings.shape} → {latent_embeddings.shape}")
    
    # Analyze original
    print("\n Analyzing original embeddings...")
    analyzer_orig = DiversityAnalyzer(min_clusters=2, max_clusters=20)
    analyzer_orig.fit_reference(embeddings)
    metrics_orig = analyzer_orig.validate_generation(embeddings)
    orig_clusters = analyzer_orig.reference_clusters
    
    # Analyze latent
    print("\n Analyzing VAE-compressed embeddings...")
    analyzer_latent = DiversityAnalyzer(min_clusters=2, max_clusters=20)
    analyzer_latent.fit_reference(latent_embeddings)
    metrics_latent = analyzer_latent.validate_generation(latent_embeddings)
    latent_clusters = analyzer_latent.reference_clusters
    
    # Visualizations
    latent_plot_path = output_dir / 'vae_latent_space.png'
    plot_latent_space(latent_embeddings, latent_clusters, latent_plot_path, 
                     title=f"VAE Latent Space ({latent_dim} dims)",
                     silhouette_score=metrics_latent.get('silhouette_score', 0),
                     diversity_score=metrics_latent.get('diversity_score', 0))
    
    comparison_plot_path = output_dir / 'clustering_comparison.png'
    plot_comparison_bar(metrics_orig, metrics_latent, comparison_plot_path)
    
    radar_plot_path = output_dir / 'metrics_radar.png'
    plot_metrics_radar(metrics_orig, metrics_latent, radar_plot_path)
    
    # Print detailed comparison
    print("\n" + "="*70)
    print(" CLUSTERING COMPARISON: Original vs VAE-Compressed")
    print("="*70)
    
    # Core metrics
    print(f"\n{' CORE METRICS':^70}")
    print("-"*70)
    print(f"{'Metric':<30} {'Original (768d)':<20} {'VAE-Compressed':<20}")
    print("-"*70)
    print(f"{'Optimal clusters':<30} {metrics_orig['optimal_clusters']:<20} {metrics_latent['optimal_clusters']:<20}")
    print(f"{'Silhouette score':<30} {metrics_orig['silhouette_score']:<20.4f} {metrics_latent['silhouette_score']:<20.4f}")
    print(f"{'Reference samples':<30} {metrics_orig['n_reference_samples']:<20} {metrics_latent['n_reference_samples']:<20}")
    
    # Diversity metrics
    print(f"\n{' DIVERSITY METRICS':^70}")
    print("-"*70)
    print(f"{'Coverage':<30} {metrics_orig.get('coverage', 0):<20.4f} {metrics_latent.get('coverage', 0):<20.4f}")
    print(f"{'Coverage %':<30} {metrics_orig.get('coverage_percentage', 0):<20.1f} {metrics_latent.get('coverage_percentage', 0):<20.1f}")
    print(f"{'Entropy ratio':<30} {metrics_orig.get('entropy_ratio', 0):<20.4f} {metrics_latent.get('entropy_ratio', 0):<20.4f}")
    print(f"{'Diversity score':<30} {metrics_orig.get('diversity_score', 0):<20.4f} {metrics_latent.get('diversity_score', 0):<20.4f}")
    
    # Quality metrics
    print(f"\n{' QUALITY METRICS':^70}")
    print("-"*70)
    print(f"{'Intra-cluster variance (ref)':<30} {metrics_orig.get('intra_cluster_variance_reference', 0):<20.6f} {metrics_latent.get('intra_cluster_variance_reference', 0):<20.6f}")
    print(f"{'Intra-cluster variance (gen)':<30} {metrics_orig.get('intra_cluster_variance_generated', 0):<20.6f} {metrics_latent.get('intra_cluster_variance_generated', 0):<20.6f}")
    print(f"{'Novelty':<30} {metrics_orig.get('novelty', 0):<20.4f} {metrics_latent.get('novelty', 0):<20.4f}")
    print(f"{'Novelty %':<30} {metrics_orig.get('novelty_percentage', 0):<20.1f} {metrics_latent.get('novelty_percentage', 0):<20.1f}")
    print(f"{'Perplexity':<30} {metrics_orig.get('perplexity', 0):<20.4f} {metrics_latent.get('perplexity', 0):<20.4f}")

    preservation = (metrics_latent['silhouette_score'] / metrics_orig['silhouette_score']) * 100
    compression_ratio = embeddings.shape[1] / latent_dim
    
    # Save results
    results = {
        'original': {
            'dim': embeddings.shape[1],
            'n_samples': len(embeddings),
            'optimal_clusters': metrics_orig['optimal_clusters'],
            'silhouette_score': metrics_orig['silhouette_score'],
            'coverage': metrics_orig.get('coverage', 0),
            'entropy_ratio': metrics_orig.get('entropy_ratio', 0),
            'diversity_score': metrics_orig.get('diversity_score', 0),
            'intra_cluster_variance': metrics_orig.get('intra_cluster_variance_reference', 0),
            'novelty': metrics_orig.get('novelty', 0),
            'perplexity': metrics_orig.get('perplexity', 0),
        },
        'vae_compressed': {
            'dim': latent_dim,
            'n_samples': len(latent_embeddings),
            'optimal_clusters': metrics_latent['optimal_clusters'],
            'silhouette_score': metrics_latent['silhouette_score'],
            'coverage': metrics_latent.get('coverage', 0),
            'entropy_ratio': metrics_latent.get('entropy_ratio', 0),
            'diversity_score': metrics_latent.get('diversity_score', 0),
            'intra_cluster_variance': metrics_latent.get('intra_cluster_variance_reference', 0),
            'novelty': metrics_latent.get('novelty', 0),
            'perplexity': metrics_latent.get('perplexity', 0),
        },
        'preservation_percent': float(preservation),
        'compression_ratio': float(compression_ratio),
        'latent_dim': latent_dim,
    }
    
    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n All results saved to: {output_dir}")
    print(f"   - Latent space plot: {latent_plot_path}")
    print(f"   - Comparison bar chart: {comparison_plot_path}")
    print(f"   - Radar chart: {radar_plot_path}")
    print(f"   - JSON results: {output_dir / 'comparison_results.json'}")
    
    return metrics_orig, metrics_latent


def main():
    parser = argparse.ArgumentParser(description="Compare original vs VAE-compressed embeddings")
    parser.add_argument('--audio_dir', help='Directory with audio files (will extract embeddings)')
    parser.add_argument('--embeddings_path', help='Path to pre-extracted embeddings .npy file')
    parser.add_argument('--vae_model', required=True, help='Path to VAE model checkpoint')
    parser.add_argument('--backend', default='wav2vec2_music', 
                        choices=['librosa', 'wav2vec2', 'wav2vec2_music'])
    parser.add_argument('--output_dir', default='outputs/vae_comparison')
    parser.add_argument('--sample_rate', type=int, default=22050)
    parser.add_argument('--duration', type=float, default=10.0)
    
    args = parser.parse_args()
    
    # Get embeddings
    if args.embeddings_path:
        print(f"\n Loading embeddings from {args.embeddings_path}...")
        embeddings = np.load(args.embeddings_path)
        print(f"   Loaded {len(embeddings)} embeddings, dim={embeddings.shape[1]}")
    elif args.audio_dir:
        print(f"\n Extracting embeddings from {args.audio_dir}...")
        extractor = AudioFeatureExtractor(backend=args.backend, duration=args.duration)
        embeddings, _ = extractor.extract_from_directory(args.audio_dir)
    else:
        raise ValueError("Either --audio_dir or --embeddings_path must be provided")
    
    compare_clustering(embeddings, args.vae_model, args.output_dir)


if __name__ == "__main__":
    main()