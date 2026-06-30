import json
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .extractor_clap import CLAPExtractor


def load_prompts_from_folder(folder):
    """Load prompts from generation_prompts.json in a folder."""
    prompt_file = Path(folder) / "generation_prompts.json"
    if not prompt_file.exists():
        prompt_file = Path(folder).parent / "generation_prompts.json"
    
    if prompt_file.exists():
        with open(prompt_file, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            if 'prompts' in data:
                return data['prompts']
            elif 'generation_prompts' in data:
                return data['generation_prompts']
    return None


def validate_clap_similarity(audio_dir, prompts, output_dir, duration=10.0):
    """
    Compute CLAP similarity between generated audio and prompts.
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("CLAP TEXT-AUDIO SIMILARITY")
    print("="*70)
    print(f"   Audio dir: {audio_dir}")
    print(f"   Prompts: {len(prompts)}")
    
    extractor = CLAPExtractor()
    
    # Extract audio embeddings
    print("\nExtracting audio embeddings...")
    audio_files = list(Path(audio_dir).glob("*.wav")) + list(Path(audio_dir).glob("*.flac"))
    audio_embeddings, audio_paths = extractor.extract_audio_batch(audio_files, duration)
    print(f"   Extracted {len(audio_embeddings)} audio embeddings")
    
    # Extract text embeddings
    print("\nExtracting text embeddings...")
    text_embeddings = extractor.extract_text_batch(prompts)
    print(f"   Extracted {len(text_embeddings)} text embeddings")
    
    # Compute similarities
    print("\nComputing similarities...")
    similarities = cosine_similarity(audio_embeddings, text_embeddings)
    
    # Statistics
    max_sims = np.max(similarities, axis=1)
    mean_sims = np.mean(similarities, axis=1)
    
    results = {
        'n_audio': len(audio_paths),
        'n_prompts': len(prompts),
        'mean_similarity': float(np.mean(similarities)),
        'std_similarity': float(np.std(similarities)),
        'mean_max_similarity': float(np.mean(max_sims)),
        'std_max_similarity': float(np.std(max_sims)),
        'audio_stats': [
            {
                'file': Path(p).name,
                'max_similarity': float(max_sims[i]),
                'mean_similarity': float(mean_sims[i]),
            }
            for i, p in enumerate(audio_paths)
        ],
        'prompts': prompts,
    }
    
    with open(output_dir / 'clap_similarity.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"   Mean similarity: {results['mean_similarity']:.4f}")
    print(f"   Mean max similarity: {results['mean_max_similarity']:.4f}")
    
    return results


def compare_clap_similarity(base_dir, rank8_dir, output_dir, duration=10.0):
    """Compare CLAP similarity between Base and Rank 8."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("CLAP SIMILARITY: Base vs Rank 8")
    print("="*70)
    
    # Load prompts from rank8 folder
    prompts = load_prompts_from_folder(rank8_dir)
    if prompts is None:
        print("ERROR: No prompts found in rank8 folder")
        return None, None
    
    print(f"Found {len(prompts)} prompts")
    
    base_results = validate_clap_similarity(base_dir, prompts, output_dir / 'base', duration)
    rank8_results = validate_clap_similarity(rank8_dir, prompts, output_dir / 'rank8', duration)
    
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"   Base mean max similarity: {base_results['mean_max_similarity']:.4f}")
    print(f"   Rank 8 mean max similarity: {rank8_results['mean_max_similarity']:.4f}")
    
    # Save comparison
    comparison = {
        'base': {
            'mean_max_similarity': base_results['mean_max_similarity'],
            'std_max_similarity': base_results['std_max_similarity'],
        },
        'rank8': {
            'mean_max_similarity': rank8_results['mean_max_similarity'],
            'std_max_similarity': rank8_results['std_max_similarity'],
        },
        'difference': rank8_results['mean_max_similarity'] - base_results['mean_max_similarity'],
    }
    
    with open(output_dir / 'comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    return base_results, rank8_results


def validate_all_folders(base_dir, output_dir, duration=10.0):
    """Validate all base and rank8 folders."""
    
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("CLAP VALIDATION: ALL FOLDERS")
    print("="*70)
    print(f"   Base dir: {base_dir}")
    
    # Find all prompt folders
    prompt_folders = []
    for folder in base_dir.iterdir():
        if folder.is_dir() and 'prompt' in folder.name and 'seed' in folder.name:
            prompt_folders.append(folder)
    
    print(f"Found {len(prompt_folders)} prompt folders")
    
    all_base_sims = []
    all_rank8_sims = []
    results = []
    
    for folder in prompt_folders:
        base_folder = folder / 'base_params'
        rank8_folder = folder / 'rank8_params'
        
        prompts = load_prompts_from_folder(folder)
        if prompts is None:
            continue
        
        result = {'prompt': folder.name}
        
        if base_folder.exists() and prompts:
            base_res = validate_clap_similarity(base_folder, prompts, output_dir / 'base' / folder.name, duration)
            result['base_mean_max'] = base_res['mean_max_similarity']
            all_base_sims.append(base_res['mean_max_similarity'])
        
        if rank8_folder.exists() and prompts:
            rank8_res = validate_clap_similarity(rank8_folder, prompts, output_dir / 'rank8' / folder.name, duration)
            result['rank8_mean_max'] = rank8_res['mean_max_similarity']
            all_rank8_sims.append(rank8_res['mean_max_similarity'])
        
        results.append(result)
    
    # Overall summary
    summary = {
        'n_folders': len(results),
        'base_mean': np.mean(all_base_sims) if all_base_sims else 0,
        'rank8_mean': np.mean(all_rank8_sims) if all_rank8_sims else 0,
        'base_std': np.std(all_base_sims) if all_base_sims else 0,
        'rank8_std': np.std(all_rank8_sims) if all_rank8_sims else 0,
        'folders': results,
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"   Base mean max similarity: {summary['base_mean']:.4f}")
    print(f"   Rank 8 mean max similarity: {summary['rank8_mean']:.4f}")
    print(f"   Difference: {summary['rank8_mean'] - summary['base_mean']:+.4f}")
    
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', help='Base directory with validation_set_2')
    parser.add_argument('--audio_dir', help='Directory with audio files (single folder)')
    parser.add_argument('--prompts_json', help='Path to generation_prompts.json (single folder)')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--duration', type=float, default=10.0)
    args = parser.parse_args()
    
    if args.base_dir:
        # Run on all folders
        validate_all_folders(args.base_dir, args.output_dir, args.duration)
    elif args.audio_dir and args.prompts_json:
        # Run on single folder
        with open(args.prompts_json, 'r') as f:
            prompts = json.load(f)
        if isinstance(prompts, dict) and 'prompts' in prompts:
            prompts = prompts['prompts']
        validate_clap_similarity(args.audio_dir, prompts, args.output_dir, args.duration)
    else:
        print("ERROR: Provide either --base_dir or both --audio_dir and --prompts_json")