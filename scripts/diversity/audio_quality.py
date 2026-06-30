import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')


def compute_audio_quality(audio_path, sample_rate=16000, duration=10.0):
    """Compute quality metrics for a single audio file."""
    
    try:
        audio, sr = librosa.load(audio_path, sr=sample_rate, duration=duration)
        
        if len(audio) < 1:
            return None
        
        # Dynamic range
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio**2))
        dynamic_range_db = 20 * np.log10((peak + 1e-12) / (rms + 1e-12))
        
        # Clipping
        clipping_count = np.sum(np.abs(audio) > 0.99)
        clipping_ratio = clipping_count / len(audio) if len(audio) > 0 else 0
        
        # Silence ratio (RMS < 0.001)
        silence_threshold = 0.001
        silence_count = np.sum(np.abs(audio) < silence_threshold)
        silence_ratio = silence_count / len(audio) if len(audio) > 0 else 0
        
        # DC offset
        dc_offset = np.mean(audio)
        
        # Spectral flatness (tonality)
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=audio))
        
        # Spectral centroid (brightness)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        
        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
        
        # Crest factor (peak/rms)
        crest_factor = peak / (rms + 1e-12)
        
        return {
            'file': Path(audio_path).name,
            'dynamic_range_db': float(dynamic_range_db),
            'peak_level': float(peak),
            'rms': float(rms),
            'crest_factor': float(crest_factor),
            'clipping_count': int(clipping_count),
            'clipping_ratio': float(clipping_ratio),
            'silence_ratio': float(silence_ratio),
            'dc_offset': float(dc_offset),
            'spectral_flatness': float(spectral_flatness),
            'spectral_centroid': float(spectral_centroid),
            'zero_crossing_rate': float(zcr),
        }
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def validate_audio_quality(generated_dir, output_dir='outputs/audio_quality', max_files=None):
    """Run audio quality validation."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print(" AUDIO QUALITY VALIDATION")
    print("="*70)
    print(f"   Generated dir: {generated_dir}")
    print(f"   Output dir: {output_dir}")
    
    audio_files = list(Path(generated_dir).glob("*.wav")) + list(Path(generated_dir).glob("*.flac"))
    
    if max_files:
        audio_files = audio_files[:max_files]
    
    print(f"   Found {len(audio_files)} files")
    
    results = []
    for f in tqdm(audio_files, desc="Processing audio"):
        metrics = compute_audio_quality(f)
        if metrics:
            results.append(metrics)
    
    if not results:
        print("⚠️ No valid audio files processed")
        return None
    
    # Aggregate summary
    summary = {
        'n_files': len(results),
        'files': results,
        'overall': {
            'avg_dynamic_range_db': np.mean([r['dynamic_range_db'] for r in results]),
            'std_dynamic_range_db': np.std([r['dynamic_range_db'] for r in results]),
            'avg_clipping_ratio': np.mean([r['clipping_ratio'] for r in results]),
            'max_clipping_ratio': np.max([r['clipping_ratio'] for r in results]),
            'avg_silence_ratio': np.mean([r['silence_ratio'] for r in results]),
            'avg_dc_offset': np.mean([r['dc_offset'] for r in results]),
            'avg_spectral_flatness': np.mean([r['spectral_flatness'] for r in results]),
            'avg_spectral_centroid': np.mean([r['spectral_centroid'] for r in results]),
            'avg_crest_factor': np.mean([r['crest_factor'] for r in results]),
        }
    }
    
    with open(output_dir / 'audio_quality.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print(" AUDIO QUALITY SUMMARY")
    print("="*70)
    print(f"   Files: {summary['n_files']}")
    print(f"   Avg dynamic range: {summary['overall']['avg_dynamic_range_db']:.1f} dB")
    print(f"   Avg clipping ratio: {summary['overall']['avg_clipping_ratio']:.4f}")
    print(f"   Avg silence ratio: {summary['overall']['avg_silence_ratio']:.4f}")
    print(f"   Avg DC offset: {summary['overall']['avg_dc_offset']:.6f}")
    print(f"   Avg spectral flatness: {summary['overall']['avg_spectral_flatness']:.4f}")
    
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated_dir', required=True)
    parser.add_argument('--output_dir', default='outputs/audio_quality')
    parser.add_argument('--max_files', type=int, default=None)
    args = parser.parse_args()
    
    validate_audio_quality(args.generated_dir, args.output_dir, args.max_files)