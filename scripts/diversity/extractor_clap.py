import numpy as np
import torch
import librosa
from pathlib import Path
from typing import List, Union, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class CLAPExtractor:
    """Extract CLAP embeddings for audio and text."""
    
    def __init__(self, model_name: str = "laion/clap-htsat-fused", device: str = "auto"):
        """
        Args:
            model_name: CLAP model from HuggingFace
            device: 'cuda', 'cpu', or 'auto'
        """
        from transformers import ClapModel, ClapProcessor
        
        self.device = self._get_device(device)
        
        print(f"Loading CLAP model on {self.device}...")
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.embedding_dim = 512
        print(f" CLAP ready (dim={self.embedding_dim})")
    
    def _get_device(self, device: str) -> str:
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def extract_audio_embedding(self, audio_path: Union[str, Path], duration: float = 10.0) -> Optional[np.ndarray]:
        """Extract CLAP embedding from audio file."""
        try:
            audio, sr = librosa.load(audio_path, sr=48000, duration=duration)
            
            # Pad if needed
            target_len = 48000 * int(duration)
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)))
            else:
                audio = audio[:target_len]
            
            # Используем правильный метод get_audio_features
            inputs = self.processor(audios=audio, sampling_rate=48000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                audio_embeds = self.model.get_audio_features(**inputs)
                embedding = audio_embeds.squeeze().cpu().numpy()
            
            return embedding
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def extract_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Extract CLAP embedding from text prompt."""
        try:
            # Используем правильный метод get_text_features
            inputs = self.processor(text=text, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_embeds = self.model.get_text_features(**inputs)
                embedding = text_embeds.squeeze().cpu().numpy()
            
            return embedding
            
        except Exception as e:
            print(f"Error processing text: {e}")
            return None
    
    def extract_audio_batch(self, audio_paths: List[Union[str, Path]], duration: float = 10.0) -> tuple:
        """Extract CLAP embeddings for multiple audio files."""
        embeddings = []
        paths = []
        
        for path in tqdm(audio_paths, desc="Extracting audio embeddings"):
            emb = self.extract_audio_embedding(path, duration)
            if emb is not None and len(emb) == self.embedding_dim:
                embeddings.append(emb)
                paths.append(str(path))
            else:
                print(f"   Skipping {path}: invalid embedding")
        
        if not embeddings:
            return np.array([]), []
        return np.array(embeddings), paths
    
    def extract_text_batch(self, texts: List[str]) -> np.ndarray:
        """Extract CLAP embeddings for multiple texts."""
        embeddings = []
        
        for text in tqdm(texts, desc="Extracting text embeddings"):
            emb = self.extract_text_embedding(text)
            if emb is not None and len(emb) == self.embedding_dim:
                embeddings.append(emb)
        
        return np.array(embeddings) if embeddings else np.array([])