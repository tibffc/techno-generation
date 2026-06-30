import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from tqdm import tqdm
import sys
import re

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.diversity.extractor_universal import AudioFeatureExtractor


def to_python(val):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    elif isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    elif isinstance(val, np.bool_):
        return bool(val)
    elif isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, dict):
        return {k: to_python(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [to_python(v) for v in val]
    else:
        return val


def analyze_seed_variance(
    validation_dir: str,
    output_dir: str = 'outputs/seed_variance',
    duration: float = 10.0
):
    """
    Compare seed variance between Base and Rank 8.
    
    For each prompt, collect all seeds, compute variance of embeddings across seeds.
    """
    
    validation_dir = Path(validation_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print(" SEED VARIANCE ANALYSIS")
    print("="*70)
    print(f"   Validation dir: {validation_dir}")
    print(f"   Output dir: {output_dir}")
    
    extractor = AudioFeatureExtractor(backend='wav2vec2_music', duration=duration)
    
    # Find all prompt folders
    prompt_groups = defaultdict(lambda: {'base': [], 'rank8': []})
    
    for folder in validation_dir.iterdir():
        if not folder.is_dir():
            continue
        
        match = re.search(r'compare_(prompt\d+)_seed(\d+)', folder.name)
        if not match:
            continue
        
        prompt = match.group(1)
        seed = int(match.group(2))
        
        base_folder = folder / 'base_params'
        rank8_folder = folder / 'rank8_params'
        
        if base_folder.exists():
            files = list(base_folder.glob("*.wav")) + list(base_folder.glob("*.flac"))
            for f in files:
                prompt_groups[prompt]['base'].append((seed, f))
        
        if rank8_folder.exists():
            files = list(rank8_folder.glob("*.wav")) + list(rank8_folder.glob("*.flac"))
            for f in files:
                prompt_groups[prompt]['rank8'].append((seed, f))
    
    print(f"\n Found {len(prompt_groups)} prompts")
    for prompt, data in prompt_groups.items():
        base_count = len(data['base'])
        rank8_count = len(data['rank8'])
        print(f"   {prompt}: Base={base_count} files, Rank8={rank8_count} files")
    
    # Compute variance
    results = {}
    all_variances = {'base': [], 'rank8': []}
    
    for prompt, data in prompt_groups.items():
        prompt_result = {'prompt': prompt, 'base': {}, 'rank8': {}}
        
        for model_type in ['base', 'rank8']:
            if not data[model_type]:
                continue
            
            seed_groups = defaultdict(list)
            for seed, f in data[model_type]:
                seed_groups[seed].append(f)
            
            seed_embeddings = {}
            for seed, files in seed_groups.items():
                if files:
                    emb = extractor.extract_features(files[0])
                    if emb is not None:
                        seed_embeddings[seed] = emb
            
            if len(seed_embeddings) < 2:
                prompt_result[model_type] = {'error': 'Less than 2 seeds'}
                continue
            
            seed_emb_list = list(seed_embeddings.values())
            seed_emb_array = np.array(seed_emb_list)
            
            mean_emb = np.mean(seed_emb_array, axis=0)
            distances = np.linalg.norm(seed_emb_array - mean_emb, axis=1)
            variance = np.mean(distances ** 2)
            
            prompt_result[model_type] = {
                'n_seeds': len(seed_embeddings),
                'variance': float(variance),  # уже float
                'seeds': list(seed_embeddings.keys()),
            }
            all_variances[model_type].append(float(variance))
        
        results[prompt] = prompt_result
    
    # Aggregate
    summary = {
        'prompts': results,
        'overall': {
            'base': {
                'mean_variance': float(np.mean(all_variances['base'])) if all_variances['base'] else 0.0,
                'std_variance': float(np.std(all_variances['base'])) if all_variances['base'] else 0.0,
                'n_prompts': len(all_variances['base']),
                'all_variances': [float(v) for v in all_variances['base']],
            },
            'rank8': {
                'mean_variance': float(np.mean(all_variances['rank8'])) if all_variances['rank8'] else 0.0,
                'std_variance': float(np.std(all_variances['rank8'])) if all_variances['rank8'] else 0.0,
                'n_prompts': len(all_variances['rank8']),
                'all_variances': [float(v) for v in all_variances['rank8']],
            }
        }
    }
    
    # Конвертируем всё в Python типы
    summary = to_python(summary)
    
    # Save
    with open(output_dir / 'seed_variance.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print
    print("\n" + "="*70)
    print(" SEED VARIANCE COMPARISON")
    print("="*70)
    
    base_mean = summary['overall']['base']['mean_variance']
    rank8_mean = summary['overall']['rank8']['mean_variance']
    base_n = summary['overall']['base']['n_prompts']
    rank8_n = summary['overall']['rank8']['n_prompts']
    
    print(f"\n   Base mean variance across seeds: {base_mean:.4f} (n={base_n})")
    print(f"   Rank 8 mean variance across seeds: {rank8_mean:.4f} (n={rank8_n})")
    
    if base_n > 0 and rank8_n > 0:
        if rank8_mean > base_mean:
            print(f"\n    Rank 8 shows HIGHER seed variance (+{(rank8_mean - base_mean)/base_mean*100:.1f}%)")
            print("      (more diverse across different seeds)")
        elif rank8_mean < base_mean:
            print(f"\n    Base shows HIGHER seed variance (+{(base_mean - rank8_mean)/rank8_mean*100:.1f}%)")
            print("      (more diverse across different seeds)")
        else:
            print("\n    Both models show similar seed variance")
    
    # Per-prompt breakdown
    print("\n PER-PROMPT BREAKDOWN:")
    print(f"{'Prompt':<15} {'Base variance':<18} {'Rank 8 variance':<18} {'Winner':<10}")
    print("-"*65)
    
    for prompt, data in results.items():
        base_var = data['base'].get('variance', 0.0) if 'variance' in data['base'] else 0.0
        rank8_var = data['rank8'].get('variance', 0.0) if 'variance' in data['rank8'] else 0.0
        
        if base_var > 0 and rank8_var > 0:
            winner = "Rank 8" if rank8_var > base_var else "Base"
        elif base_var > 0:
            winner = "Base"
        elif rank8_var > 0:
            winner = "Rank 8"
        else:
            winner = "N/A"
        
        print(f"{prompt:<15} {base_var:<18.4f} {rank8_var:<18.4f} {winner:<10}")
    
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation_dir', required=True)
    parser.add_argument('--output_dir', default='outputs/seed_variance')
    parser.add_argument('--duration', type=float, default=10.0)
    args = parser.parse_args()
    
    analyze_seed_variance(args.validation_dir, args.output_dir, args.duration)