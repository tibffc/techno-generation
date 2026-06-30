import numpy as np
import librosa
import torch
from pathlib import Path
from typing import List, Union, Optional, Dict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class AudioFeatureExtractor:
    """
    Universal audio feature extractor.
    
    Available backends:
        - 'librosa': 39 handcrafted features (fastest, no GPU required)
        - 'wav2vec2': 768-dim embeddings from vanilla Wav2Vec2.0 (speech-pretrained)
        - 'wav2vec2_music': 768-dim embeddings from fine-tuned Wav2Vec2 (music-pretrained)
    """
    
    def __init__(self, backend: str = 'librosa', 
                 sample_rate: int = 16000,
                 duration: float = 10.0,
                 device: str = 'auto'):
        """
        Args:
            backend: 'librosa', 'wav2vec2', or 'wav2vec2_music'
            sample_rate: Target sample rate (Wav2Vec2 expects 16000)
            duration: Duration in seconds to analyze
            device: 'cuda', 'cpu', or 'auto'
        """
        self.backend = backend
        self.duration = duration
        self.device = self._get_device(device)

        if backend == 'librosa':
            self.sample_rate = 22050  # Оптимально для librosa
        else:
            self.sample_rate = 16000  # Wav2Vec2 требует 16kHz
        
        self.n_samples = int(self.sample_rate * duration)
        
        self.model = None
        self.processor = None
        self.embedding_dim = None
        self._load_model()
        
    def _get_device(self, device: str) -> str:
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _load_model(self):
        """Load the appropriate model based on backend selection."""
        
        if self.backend == 'librosa':
            print(f" Using Librosa (handcrafted features)")
            self.embedding_dim = 39
            return
        
        elif self.backend == 'wav2vec2':
            try:
                from transformers import Wav2Vec2Processor, Wav2Vec2Model
                print(f"Loading vanilla Wav2Vec2 on {self.device}...")
                self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
                self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
                self.model = self.model.to(self.device)
                self.model.eval()
                self.embedding_dim = 768
                print(f" Vanilla Wav2Vec2 ready (dim={self.embedding_dim})")
            except Exception as e:
                print(f" Wav2Vec2 failed: {e}")
                raise
        
        elif self.backend == 'wav2vec2_music':
            try:
                from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
                print(f"Loading fine-tuned Wav2Vec2 (music) on {self.device}...")
                model_name = "m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres"
                self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
                self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
                self.model = self.model.to(self.device)
                self.model.eval()
                self.embedding_dim = 768
                print(f" Fine-tuned Wav2Vec2 (music) ready (dim={self.embedding_dim})")
            except Exception as e:
                print(f" Wav2Vec2 music model failed: {e}")
                raise
        
        else:
            raise ValueError(f"Unknown backend: {self.backend}. Choose: 'librosa', 'wav2vec2', 'wav2vec2_music'")
    
    def _extract_librosa(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract handcrafted features using librosa (enhanced version)."""
        features = []
        
        # ========== 1. TEMPO & RHYTHM ==========
        # Tempo (BPM)
        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            features.append(float(tempo) if isinstance(tempo, (int, float)) else float(tempo[0]))
        except:
            features.append(120.0)
        
        # Onset density
        try:
            onset = librosa.onset.onset_detect(y=audio, sr=sr)
            features.append(len(onset) / self.duration)
        except:
            features.append(0.0)
        
        # Onset strength mean and std
        try:
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            features.append(float(np.mean(onset_env)))
            features.append(float(np.std(onset_env)))
        except:
            features.extend([0.0, 0.0])
        
        # Tempo confidence (from autocorrelation)
        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr, units='time')
            features.append(1.0)  # placeholder for confidence
        except:
            features.append(0.0)
        
        # ========== 2. SPECTRAL FEATURES ==========
        # Spectral centroid (brightness)
        try:
            cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features.append(float(np.mean(cent)))
            features.append(float(np.std(cent)))  # added std
        except:
            features.extend([1000.0, 0.0])
        
        # Spectral bandwidth
        try:
            bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            features.append(float(np.mean(bw)))
            features.append(float(np.std(bw)))  # added std
        except:
            features.extend([500.0, 0.0])
        
        # Spectral rolloff
        try:
            roll = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            features.append(float(np.mean(roll)))
        except:
            features.append(5000.0)
        
        # Spectral flatness (noise-like vs tone-like)
        try:
            flatness = librosa.feature.spectral_flatness(y=audio)
            features.append(float(np.mean(flatness)))
        except:
            features.append(0.5)
        
        # ========== 3. ENERGY & DYNAMICS ==========
        # RMS energy
        try:
            rms = librosa.feature.rms(y=audio)
            features.append(float(np.mean(rms)))
            features.append(float(np.std(rms)))  # dynamics variation
        except:
            features.extend([0.1, 0.0])
        
        # Zero crossing rate
        try:
            zcr = librosa.feature.zero_crossing_rate(y=audio)
            features.append(float(np.mean(zcr)))
        except:
            features.append(0.05)
        
        # ========== 4. HARMONIC & PERCUSSIVE ==========
        # Harmonic and percussive components
        try:
            harmonic, percussive = librosa.effects.hpss(audio)
            harmonic_energy = np.mean(harmonic ** 2)
            percussive_energy = np.mean(percussive ** 2)
            features.append(float(harmonic_energy))
            features.append(float(percussive_energy))
            features.append(float(harmonic_energy / (percussive_energy + 1e-8)))  # ratio
        except:
            features.extend([0.0, 0.0, 0.0])
        
        # ========== 5. MFCC (13 coefficients + deltas) ==========
        try:
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            features.extend(np.mean(mfcc, axis=1).tolist())
            # Add MFCC deltas (temporal dynamics)
            mfcc_delta = librosa.feature.delta(mfcc)
            features.extend(np.mean(mfcc_delta, axis=1).tolist())
        except:
            features.extend([0.0] * 26)
        
        # ========== 6. CHROMA (12 bins) ==========
        try:
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features.extend(np.mean(chroma, axis=1).tolist())
        except:
            features.extend([0.0] * 12)
        
        # ========== 7. SPECTRAL CONTRAST (7 bands) ==========
        try:
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            features.extend(np.mean(contrast, axis=1).tolist())
        except:
            features.extend([0.0] * 7)
        
        # ========== 8. TONNETZ (6-dimensional tonality) ==========
        try:
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
            features.extend(np.mean(tonnetz, axis=1).tolist())
        except:
            features.extend([0.0] * 6)
        
        # ========== 9. ADDITIONAL FEATURES ==========
        # Poly features (zero crossing, etc.)
        try:
            poly = librosa.feature.poly_features(y=audio, sr=sr, order=2)
            features.append(float(np.mean(poly[0])))  # coefficient a
            features.append(float(np.mean(poly[1])))  # coefficient b
        except:
            features.extend([0.0, 0.0])
        
        # Perceptual loudness (approximation)
        try:
            loudness = 20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-8)
            features.append(float(np.clip(loudness, -60, 0)))
        except:
            features.append(-20.0)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_wav2vec2(self, audio: np.ndarray, sr: int, use_music_model: bool = False) -> np.ndarray:
        """
        Extract embeddings from Wav2Vec2 model.
        
        Args:
            audio: Audio array
            sr: Sample rate of audio
            use_music_model: If True, use fine-tuned music model; else use vanilla
        """
        # Resample to 16kHz if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        # Pad or truncate to exact length
        target_len = 16000 * int(self.duration)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        
        # Process with processor
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Different models have different output structures
            if hasattr(outputs, 'last_hidden_state'):
                # Vanilla Wav2Vec2Model
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
            elif hasattr(outputs, 'logits'):
                # Fine-tuned model (has classification head)
                # We want the embeddings before classification
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    embedding = outputs.hidden_states[-1].mean(dim=1).cpu().numpy().squeeze()
                else:
                    # Fallback: use the wav2vec2 submodule
                    wav2vec2_outputs = self.model.wav2vec2(**inputs)
                    embedding = wav2vec2_outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
            else:
                # Fallback: try to get from model's wav2vec2 component
                wav2vec2_outputs = self.model.wav2vec2(**inputs)
                embedding = wav2vec2_outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
        
        return embedding
    
    def extract_features(self, audio_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Extract features/embeddings from a single audio file.
        
        Returns:
            np.ndarray of features or None if error
        """
        try:
            # Load audio at appropriate sample rate
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            
            # Pad or truncate to exact length
            target_len = self.sample_rate * int(self.duration)
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)))
            else:
                audio = audio[:target_len]
            
            # Extract based on backend
            if self.backend == 'librosa':
                return self._extract_librosa(audio, sr)
            elif self.backend == 'wav2vec2':
                return self._extract_wav2vec2(audio, sr, use_music_model=False)
            elif self.backend == 'wav2vec2_music':
                return self._extract_wav2vec2(audio, sr, use_music_model=True)
            else:
                raise ValueError(f"Unknown backend: {self.backend}")
                
        except Exception as e:
            print(f"Error processing {Path(audio_path).name}: {e}")
            return None
    
    def extract_batch(self, audio_paths: List[Union[str, Path]], 
                      progress: bool = True) -> tuple[np.ndarray, List[str]]:
        """
        Extract features for multiple audio files.
        
        Returns:
            features: np.ndarray of shape (n_files, feature_dim)
            valid_paths: list of paths that were successfully processed
        """
        features_list = []
        valid_paths = []
        
        iterator = tqdm(audio_paths, desc=f"Extracting ({self.backend})") if progress else audio_paths
        
        for path in iterator:
            feats = self.extract_features(path)
            if feats is not None:
                features_list.append(feats)
                valid_paths.append(str(path))
        
        if not features_list:
            raise ValueError(f"No valid features extracted from {len(audio_paths)} files")
        
        return np.array(features_list), valid_paths
    
    def extract_from_directory(self, directory: Union[str, Path],
                                extensions: List[str] = ['.wav', '.flac', '.mp3', '.m4a'],
                                progress: bool = True) -> tuple[np.ndarray, List[str]]:
        """
        Extract features from all audio files in a directory.
        
        Args:
            directory: Path to directory containing audio files
            extensions: List of audio extensions to process
            progress: Show progress bar
            
        Returns:
            features: np.ndarray of shape (n_files, feature_dim)
            paths: list of file paths
        """
        directory = Path(directory)
        audio_files = []
        for ext in extensions:
            audio_files.extend(directory.glob(f"*{ext}"))
        
        if not audio_files:
            raise ValueError(f"No audio files with extensions {extensions} found in {directory}")
        
        print(f"Found {len(audio_files)} audio files in {directory}")
        return self.extract_batch(audio_files, progress=progress)
    
    def get_info(self) -> Dict:
        """Return information about the extractor."""
        return {
            'backend': self.backend,
            'device': self.device,
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'embedding_dim': self.embedding_dim,
        }


# Quick test when run directly
if __name__ == "__main__":
    import sys
    print("="*60)
    print("Testing AudioFeatureExtractor")
    print("="*60)
    
    # Test Librosa
    print("\n1. Testing Librosa backend...")
    e1 = AudioFeatureExtractor(backend='librosa', duration=5.0)
    print(f"   Info: {e1.get_info()}")
    
    # Test Wav2Vec2 vanilla
    print("\n2. Testing Wav2Vec2 vanilla backend...")
    e2 = AudioFeatureExtractor(backend='wav2vec2', duration=5.0)
    print(f"   Info: {e2.get_info()}")
    
    # Test Wav2Vec2 music
    print("\n3. Testing Wav2Vec2 music backend...")
    e3 = AudioFeatureExtractor(backend='wav2vec2_music', duration=5.0)
    print(f"   Info: {e3.get_info()}")
    