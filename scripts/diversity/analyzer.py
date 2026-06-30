import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import entropy
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import ConvexHull
from typing import Dict, Optional, List, Tuple
import warnings
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')


class DiversityAnalyzer:
    """Analyze diversity of audio samples with automatic cluster detection."""
    
    def __init__(self, min_clusters: int = 2, max_clusters: int = 20, 
                 method: str = 'silhouette', random_state: int = 42,
                 cache_path: str = None):
        """
        Args:
            min_clusters: Minimum number of clusters to test
            max_clusters: Maximum number of clusters to test
            method: 'silhouette' or 'elbow' — method for optimal clusters
            random_state: Random seed for reproducibility
            cache_path: Path to cache fitted reference (e.g., 'cache/reference.pkl')
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.method = method
        self.random_state = random_state
        self.cache_path = Path(cache_path) if cache_path else None
        
        self.scaler = StandardScaler()
        self.kmeans = None
        self.n_clusters_optimal = None
        self.reference_features = None
        self.reference_clusters = None
        self.silhouette = None
        self.intra_cluster_variances = None
        self.centroids = None
        self.ref_counts = None
        self.ref_dist = None
        self.ref_entropy = None
        
    def _find_optimal_clusters(self, features: np.ndarray) -> int:
        """Automatically find optimal number of clusters."""
        if len(features) < self.min_clusters:
            return max(2, len(features) // 3)
        
        inertias = []
        silhouette_scores = []
        
        K_range = range(self.min_clusters, min(self.max_clusters + 1, len(features) - 1))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(features)
            inertias.append(kmeans.inertia_)
            
            if k >= 2 and len(set(labels)) > 1:
                try:
                    score = silhouette_score(features, labels)
                    silhouette_scores.append(score)
                except:
                    silhouette_scores.append(-1)
            else:
                silhouette_scores.append(-1)
        
        if self.method == 'silhouette' and silhouette_scores:
            best_idx = np.argmax(silhouette_scores)
            return K_range[best_idx]
        else:
            if len(inertias) < 3:
                return K_range[-1]
            diffs = np.diff(inertias)
            diffs2 = np.diff(diffs)
            elbow_idx = np.argmax(diffs2) + 1 if len(diffs2) > 0 else len(inertias) // 2
            return K_range[min(elbow_idx, len(K_range) - 1)]
    
    def fit_reference(self, features: np.ndarray, force_recompute: bool = False) -> dict:
        """Fit clustering on reference dataset with automatic cluster count.
        
        Args:
            features: Reference embeddings
            force_recompute: If True, ignore cache and recompute
        """
        # Check cache
        if not force_recompute and self.cache_path and self.cache_path.exists():
            print(f"   Loading cached reference from {self.cache_path}...")
            try:
                with open(self.cache_path, 'rb') as f:
                    cached = pickle.load(f)
                self.scaler = cached['scaler']
                self.kmeans = cached['kmeans']
                self.n_clusters_optimal = cached['n_clusters_optimal']
                self.reference_features = cached['reference_features']
                self.reference_clusters = cached['reference_clusters']
                self.silhouette = cached['silhouette']
                self.intra_cluster_variances = cached['intra_cluster_variances']
                self.centroids = cached['centroids']
                self.ref_counts = cached['ref_counts']
                self.ref_dist = cached['ref_dist']
                self.ref_entropy = cached['ref_entropy']
                print(f"   Loaded cached reference (clusters={self.n_clusters_optimal}, silhouette={self.silhouette:.4f})")
                return {
                    'optimal_clusters': self.n_clusters_optimal,
                    'silhouette_score': float(self.silhouette),
                    'tested_range': [self.min_clusters, self.max_clusters],
                    'intra_cluster_variance_reference': float(np.mean(self.intra_cluster_variances)),
                    'intra_cluster_variances_by_cluster': [float(v) for v in self.intra_cluster_variances],
                    'cached': True
                }
            except Exception as e:
                print(f"   Cache load failed: {e}, recomputing...")
        
        # Normalize
        print(f"   Normalizing {len(features)} reference samples...")
        self.reference_features = self.scaler.fit_transform(features)
        
        # Find optimal clusters
        print(f"   Finding optimal clusters (range {self.min_clusters}-{self.max_clusters})...")
        self.n_clusters_optimal = self._find_optimal_clusters(self.reference_features)
        print(f"    Auto-detected optimal clusters: {self.n_clusters_optimal}")
        
        # Fit KMeans
        self.kmeans = KMeans(n_clusters=self.n_clusters_optimal, 
                              random_state=self.random_state, 
                              n_init=10)
        self.reference_clusters = self.kmeans.fit_predict(self.reference_features)
        self.centroids = self.kmeans.cluster_centers_
        
        # Store reference distribution
        self.ref_counts = np.bincount(self.reference_clusters, minlength=self.n_clusters_optimal)
        self.ref_dist = self.ref_counts / self.ref_counts.sum() if self.ref_counts.sum() > 0 else np.zeros_like(self.ref_counts)
        self.ref_entropy = entropy(self.ref_dist[self.ref_dist > 0]) if np.any(self.ref_dist > 0) else 0
        
        # Calculate silhouette
        if len(set(self.reference_clusters)) > 1:
            self.silhouette = silhouette_score(self.reference_features, self.reference_clusters)
        else:
            self.silhouette = 0.0
        
        # Calculate intra-cluster variance
        self.intra_cluster_variances = []
        for i in range(self.n_clusters_optimal):
            cluster_features = self.reference_features[self.reference_clusters == i]
            if len(cluster_features) > 1:
                var = np.var(cluster_features, axis=0).mean()
            else:
                var = 0.0
            self.intra_cluster_variances.append(var)
        
        # Save to cache
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump({
                    'scaler': self.scaler,
                    'kmeans': self.kmeans,
                    'n_clusters_optimal': self.n_clusters_optimal,
                    'reference_features': self.reference_features,
                    'reference_clusters': self.reference_clusters,
                    'silhouette': self.silhouette,
                    'intra_cluster_variances': self.intra_cluster_variances,
                    'centroids': self.centroids,
                    'ref_counts': self.ref_counts,
                    'ref_dist': self.ref_dist,
                    'ref_entropy': self.ref_entropy,
                }, f)
            print(f"   Cached reference to {self.cache_path}")
        
        return {
            'optimal_clusters': self.n_clusters_optimal,
            'silhouette_score': float(self.silhouette),
            'tested_range': [self.min_clusters, self.max_clusters],
            'intra_cluster_variance_reference': float(np.mean(self.intra_cluster_variances)),
            'intra_cluster_variances_by_cluster': [float(v) for v in self.intra_cluster_variances],
            'cached': False
        }
    
    def compute_creativity_metrics(self, generated_features: np.ndarray, 
                                    gen_clusters: np.ndarray = None) -> dict:
        """
        Compute creativity-oriented metrics:
        - CTR: Cluster Transition Rate (гибридность)
        - S-score: Surprise (неожиданность)
        - NCI: Novel Combination Index (редкость комбинаций)
        - CLD: Cluster Leap Distance (резкость переходов)
        - Intra-cluster richness (разнообразие внутри кластера)
        - Cross-cluster novelty (новизна на границах)
        """
        if self.kmeans is None:
            raise ValueError("Must call fit_reference first")
        
        gen_scaled = self.scaler.transform(generated_features)
        if gen_clusters is None:
            gen_clusters = self.kmeans.predict(gen_scaled)
        
        centroids = self.centroids
        
        # 1. CTR - Cluster Transition Rate (гибридность)
        distances_to_centroids = cdist(gen_scaled, centroids)
        sorted_distances = np.sort(distances_to_centroids, axis=1)
        ctr_raw = sorted_distances[:, 1] / (sorted_distances[:, 0] + 1e-8)
        ctr = float(np.mean(ctr_raw))
        hybrid_percent = float(np.sum(ctr_raw > 1.5) / len(ctr_raw) * 100)
        
        # Extreme hybrids (higher threshold)
        extreme_hybrid_percent = float(np.sum(ctr_raw > 2.5) / len(ctr_raw) * 100)
        
        # 2. S-score - Surprise (неожиданность)
        cluster_centers = centroids[gen_clusters]
        distances_to_center = np.linalg.norm(gen_scaled - cluster_centers, axis=1)
        intra_cluster_stds = []
        for i in range(self.n_clusters_optimal):
            cluster_points = gen_scaled[gen_clusters == i]
            if len(cluster_points) > 1:
                intra_cluster_stds.append(np.std(cluster_points, axis=0).mean())
            else:
                intra_cluster_stds.append(1.0)
        expected_std = np.mean(intra_cluster_stds)
        surprise_raw = distances_to_center / (expected_std + 1e-8)
        s_score = float(np.mean(surprise_raw))
        high_surprise_percent = float(np.sum(surprise_raw > 2.0) / len(surprise_raw) * 100)
        
        # 3. Intra-cluster richness (разнообразие внутри кластера)
        intra_cluster_richness = float(np.var(gen_scaled, axis=0).mean())
        
        # 4. Cross-cluster novelty (новизна на границах)
        # Процент треков, далёких от всех центроидов (>95 перцентиля)
        min_distances = np.min(distances_to_centroids, axis=1)
        percentile_95 = np.percentile(min_distances, 95)
        cross_cluster_novelty = float(np.sum(min_distances > percentile_95) / len(min_distances) * 100)
        
        # 5. NCI - Novel Combination Index (редкость комбинаций)
        n_dims = gen_scaled.shape[1]
        nci_scores = []
        for track in gen_scaled[:min(1000, len(gen_scaled))]:
            track_rarity = 0
            for dim in range(min(n_dims, 100)):  # ограничим для скорости
                percentile = np.sum(self.reference_features[:, dim] < track[dim]) / len(self.reference_features)
                if percentile < 0.05 or percentile > 0.95:
                    track_rarity += 1
            nci_scores.append(track_rarity / min(n_dims, 100))
        nci = float(np.mean(nci_scores)) if nci_scores else 0
        rare_combinations_percent = float(np.sum(np.array(nci_scores) > 0.3) / len(nci_scores) * 100) if nci_scores else 0
        
        # 6. CLD - Cluster Leap Distance (резкость переходов)
        if len(gen_clusters) > 1:
            cluster_transitions = []
            for i in range(len(gen_clusters) - 1):
                c1, c2 = gen_clusters[i], gen_clusters[i+1]
                if c1 != c2:
                    dist = np.linalg.norm(centroids[c1] - centroids[c2])
                    cluster_transitions.append(dist)
                else:
                    cluster_transitions.append(0.0)
            cld = float(np.mean(cluster_transitions))
            transition_rate = float(np.sum(np.diff(gen_clusters) != 0) / (len(gen_clusters) - 1) * 100)
        else:
            cld = 0.0
            transition_rate = 0.0
        
        # 7. Creativity Index (составная метрика)
        # Нормализуем компоненты
        s_score_norm = min(s_score / 3.0, 1.0)
        ctr_norm = min(ctr / 20.0, 1.0)
        richness_norm = min(intra_cluster_richness / 0.2, 1.0)
        creativity_index = (s_score_norm * 0.4) + (ctr_norm * 0.3) + (richness_norm * 0.3)
        
        # 8. Surprise-to-Coverage Ratio
        coverage = np.sum(np.bincount(gen_clusters, minlength=self.n_clusters_optimal) > 0) / self.n_clusters_optimal
        scr = s_score / (coverage + 0.01)
        
        return {
            # CTR metrics
            'ctr': ctr,
            'hybrid_percent': hybrid_percent,
            'extreme_hybrid_percent': extreme_hybrid_percent,
            
            # S-score metrics
            's_score': s_score,
            'high_surprise_percent': high_surprise_percent,
            
            # Intra-cluster richness
            'intra_cluster_richness': intra_cluster_richness,
            
            # Cross-cluster novelty
            'cross_cluster_novelty': cross_cluster_novelty,
            
            # NCI metrics
            'nci': nci,
            'rare_combinations_percent': rare_combinations_percent,
            
            # CLD metrics
            'cld': cld,
            'transition_rate': transition_rate,
            
            # Composite metrics
            'creativity_index': creativity_index,
            'surprise_to_coverage_ratio': scr,
        }
    
    def validate_generation(self, generated_features: np.ndarray) -> Dict:
        """Compare generated samples to reference distribution."""
        if self.kmeans is None:
            raise ValueError("Must call fit_reference first")
        
        # Normalize and assign clusters
        gen_scaled = self.scaler.transform(generated_features)
        gen_clusters = self.kmeans.predict(gen_scaled)
        
        # Compute metrics
        gen_counts = np.bincount(gen_clusters, minlength=self.n_clusters_optimal)
        gen_dist = gen_counts / gen_counts.sum() if gen_counts.sum() > 0 else np.zeros_like(gen_counts)
        gen_entropy = entropy(gen_dist[gen_dist > 0]) if np.any(gen_dist > 0) else 0
        
        # Coverage
        coverage = np.sum(gen_counts > 0) / self.n_clusters_optimal
        
        # Entropy ratio
        entropy_ratio = gen_entropy / self.ref_entropy if self.ref_entropy > 0 else 1.0
        
        # Diversity score
        diversity_score = coverage * entropy_ratio
        
        # Novelty
        threshold = 0.05
        if len(generated_features) > 0:
            rare_clusters = self.ref_counts < (len(self.reference_features) * threshold)
            novelty = np.sum(gen_counts[rare_clusters]) / len(generated_features)
        else:
            novelty = 0.0
        
        # Perplexity
        distances = []
        for g in gen_scaled:
            centroid_distances = cdist([g], self.centroids).flatten()
            min_dist = np.min(centroid_distances)
            distances.append(min_dist)
        perplexity = float(np.mean(distances)) if distances else 0.0
        
        # Intra-cluster variance for generated samples
        gen_intra_variances = []
        for i in range(self.n_clusters_optimal):
            cluster_features = gen_scaled[gen_clusters == i]
            if len(cluster_features) > 1:
                var = np.var(cluster_features, axis=0).mean()
            else:
                var = 0.0
            gen_intra_variances.append(var)
        gen_intra_variance = float(np.mean(gen_intra_variances)) if gen_intra_variances else 0.0
        
        # Find under/over represented clusters
        underrepresented = []
        overrepresented = []
        
        for i in range(self.n_clusters_optimal):
            ref_pct = self.ref_counts[i] / self.ref_counts.sum() if self.ref_counts.sum() > 0 else 0
            gen_pct = gen_counts[i] / gen_counts.sum() if gen_counts.sum() > 0 else 0
            
            if ref_pct > 0:
                ratio = gen_pct / ref_pct
                
                if ratio < 0.3:
                    underrepresented.append({
                        'cluster': int(i),
                        'reference_pct': float(ref_pct),
                        'generated_pct': float(gen_pct),
                        'ratio': float(ratio),
                        'reference_count': int(self.ref_counts[i]),
                        'generated_count': int(gen_counts[i])
                    })
                elif ratio > 2.0:
                    overrepresented.append({
                        'cluster': int(i),
                        'reference_pct': float(ref_pct),
                        'generated_pct': float(gen_pct),
                        'ratio': float(ratio),
                        'reference_count': int(self.ref_counts[i]),
                        'generated_count': int(gen_counts[i])
                    })
        
        metrics = {
            # Core metrics
            'optimal_clusters': self.n_clusters_optimal,
            'silhouette_score': float(self.silhouette),
            'n_reference_samples': len(self.reference_features),
            'n_generated_samples': len(generated_features),
            
            # Diversity metrics
            'coverage': float(coverage),
            'coverage_percentage': float(coverage * 100),
            'entropy_reference': float(self.ref_entropy),
            'entropy_generated': float(gen_entropy),
            'entropy_ratio': float(entropy_ratio),
            'diversity_score': float(diversity_score),
            
            # Quality metrics
            'intra_cluster_variance_reference': float(np.mean(self.intra_cluster_variances)),
            'intra_cluster_variance_generated': gen_intra_variance,
            'novelty': float(novelty),
            'novelty_percentage': float(novelty * 100),
            'perplexity': float(perplexity),
            
            # Detailed analysis
            'overrepresented_clusters': overrepresented,
            'underrepresented_clusters': underrepresented,
            'cluster_distribution': {
                'reference': self.ref_counts.tolist(),
                'generated': gen_counts.tolist()
            }
        }
        
        # Добавляем креативные метрики (всегда, не только при условии)
        print("   Computing creativity metrics...")
        creativity_metrics = self.compute_creativity_metrics(generated_features, gen_clusters)
        metrics['creativity'] = creativity_metrics
        
        return metrics
    
    def get_cluster_centroids(self) -> np.ndarray:
        return self.scaler.inverse_transform(self.centroids)
    
    def get_cluster_labels(self, features: np.ndarray) -> np.ndarray:
        if self.kmeans is None:
            raise ValueError("Must call fit_reference first")
        scaled = self.scaler.transform(features)
        return self.kmeans.predict(scaled)
    
    def get_cluster_distribution(self) -> Dict:
        return {
            'counts': self.ref_counts.tolist(),
            'percentages': (self.ref_counts / self.ref_counts.sum() * 100).tolist(),
            'entropy': self.ref_entropy,
        }