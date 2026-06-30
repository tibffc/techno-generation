"""
Validate diversity using VAE-compressed embeddings.

Usage:
    python scripts/validate_with_vae.py \
        --reference_embeddings data/embeddings_full.npy \
        --generated_dir data/gen_clean \
        --vae_model outputs/vae_model_full/best_model.pt \
        --output_dir outputs/validation_vae
"""

import argparse
import sys
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.diversity.extractor_universal import AudioFeatureExtractor
from scripts.diversity.analyzer import DiversityAnalyzer
from scripts.diversity.vae_trainer import VAE


def compress_embeddings(embeddings_768, model, device, batch_size=128):
    """Compress 768-dim embeddings to latent space using VAE."""
    print(f"   Compressing {len(embeddings_768)} embeddings...")
    compressed = []
    with torch.no_grad():
        for i in range(0, len(embeddings_768), batch_size):
            batch = torch.FloatTensor(embeddings_768[i:i+batch_size]).to(device)
            mu, _ = model.encode(batch)
            compressed.append(mu.cpu().numpy())
    return np.concatenate(compressed, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Validate diversity with VAE compression")
    parser.add_argument('--reference_embeddings', required=True,
                        help='Path to pre-extracted reference embeddings (.npy)')
    parser.add_argument('--generated_dir', required=True,
                        help='Directory with generated samples')
    parser.add_argument('--vae_model', required=True,
                        help='Path to VAE model checkpoint (.pt)')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for report')
    parser.add_argument('--min_clusters', type=int, default=2)
    parser.add_argument('--max_clusters', type=int, default=20)
    parser.add_argument('--duration', type=float, default=10.0)
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print(" DIVERSITY VALIDATION WITH VAE")
    print("="*60)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n Device: {device}")
    
    # 1. Load VAE model
    print(f"\n Loading VAE model from {args.vae_model}...")
    checkpoint = torch.load(args.vae_model, map_location=device)
    latent_dim = checkpoint['latent_dim']
    model = VAE(input_dim=768, latent_dim=latent_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"   VAE loaded (latent_dim={latent_dim})")
    
    # 2. Load reference embeddings (768d)
    print(f"\n Loading reference embeddings from {args.reference_embeddings}...")
    ref_embeddings_768 = np.load(args.reference_embeddings)
    print(f"   Loaded {len(ref_embeddings_768)} embeddings, dim=768")
    
    # 3. Compress reference embeddings
    print(f"\n Compressing reference embeddings...")
    ref_embeddings = compress_embeddings(ref_embeddings_768, model, device)
    print(f"   Compressed to {ref_embeddings.shape}")
    
    # 4. Extract embeddings from generated samples
    print(f"\n Extracting embeddings from generated: {args.generated_dir}")
    extractor = AudioFeatureExtractor(backend='wav2vec2_music', duration=args.duration)
    gen_embeddings_768, gen_paths = extractor.extract_from_directory(args.generated_dir)
    print(f"   Extracted {len(gen_embeddings_768)} generated embeddings, dim=768")
    
    # 5. Compress generated embeddings
    print(f"\n Compressing generated embeddings...")
    gen_embeddings = compress_embeddings(gen_embeddings_768, model, device)
    print(f"   Compressed to {gen_embeddings.shape}")
    
    # 6. Run clustering and validation
    print(f"\n Running clustering and validation...")
    analyzer = DiversityAnalyzer(min_clusters=args.min_clusters, max_clusters=args.max_clusters)
    analyzer.fit_reference(ref_embeddings)
    metrics = analyzer.validate_generation(gen_embeddings)
    
    # 7. Add metadata
    metrics['backend'] = 'vae'
    metrics['latent_dim'] = latent_dim
    metrics['reference_embeddings_path'] = str(args.reference_embeddings)
    metrics['generated_dir'] = str(args.generated_dir)
    
    # 8. Save report
    output_path = Path(args.output_dir) / 'diversity_report.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n Report saved to {output_path}")
    
    # 9. Print summary
    print("\n" + "="*60)
    print(" DIVERSITY VALIDATION SUMMARY (VAE)")
    print("="*60)
    print(f"Reference samples: {metrics['n_reference_samples']}")
    print(f"Generated samples: {metrics['n_generated_samples']}")
    print(f"Optimal clusters: {metrics['optimal_clusters']}")
    print(f"Silhouette score: {metrics['silhouette_score']:.4f}")
    
    print("\n DIVERSITY METRICS:")
    print(f"   Coverage: {metrics['coverage_percentage']:.1f}%")
    print(f"   Entropy ratio: {metrics['entropy_ratio']:.3f}")
    print(f"   Diversity score: {metrics['diversity_score']:.3f}")
    
    print("\n QUALITY METRICS:")
    print(f"   Intra-cluster variance (ref): {metrics['intra_cluster_variance_reference']:.6f}")
    print(f"   Intra-cluster variance (gen): {metrics['intra_cluster_variance_generated']:.6f}")
    print(f"   Novelty: {metrics['novelty_percentage']:.1f}%")
    print(f"   Perplexity: {metrics['perplexity']:.4f}")
    
    if metrics.get('underrepresented_clusters'):
        print(f"\n UNDERREPRESENTED CLUSTERS ({len(metrics['underrepresented_clusters'])}):")
        for c in metrics['underrepresented_clusters'][:5]:
            print(f"   Cluster {c['cluster']}: {c['generated_pct']*100:.1f}% vs ref {c['reference_pct']*100:.1f}%")
    
    print("="*60)
    
    # 10. Compare with original (if we have previous result)
    orig_report = Path(args.output_dir).parent / 'validation_gen_clean' / 'diversity_report.json'
    if orig_report.exists():
        with open(orig_report) as f:
            orig_metrics = json.load(f)
        print("\n COMPARISON: Original vs VAE")
        print("-"*40)
        print(f"Coverage:       {orig_metrics['coverage_percentage']:.1f}% → {metrics['coverage_percentage']:.1f}%")
        print(f"Diversity:      {orig_metrics['diversity_score']:.3f} → {metrics['diversity_score']:.3f}")
        print(f"Entropy ratio:  {orig_metrics['entropy_ratio']:.3f} → {metrics['entropy_ratio']:.3f}")
        print(f"Perplexity:     {orig_metrics['perplexity']:.4f} → {metrics['perplexity']:.4f}")
        
        if metrics['diversity_score'] > orig_metrics['diversity_score']:
            print("\n VAE shows IMPROVED diversity metrics!")
        else:
            print("\n VAE shows similar or slightly lower diversity metrics.")


if __name__ == "__main__":
    main()