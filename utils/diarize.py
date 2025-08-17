from typing import List
import numpy as np
import librosa
from sklearn.cluster import AgglomerativeClustering

def _segment_mfcc_embedding(wav_path: str, start: float, end: float, sr: int = 16000) -> np.ndarray:
    y, sr = librosa.load(wav_path, sr=sr, offset=start, duration=max(0.05, end - start))
    if y.size == 0:
        return np.zeros(13, dtype=float)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

def diarize_by_clustering(wav_path: str, segments: List[dict]) -> List[str]:
    """Lightweight diarization: cluster MFCC means → speaker labels.

    Returns list of speaker labels aligned with input segments.

    Uses distance threshold to infer number of speakers automatically.

    """
    if not segments:
        return []
    X = []
    for seg in segments:
        X.append(_segment_mfcc_embedding(wav_path, float(seg['start']), float(seg['end'])))
    X = np.vstack(X)

    # If there's only one segment, just one speaker
    if len(segments) == 1:
        return ["SPEAKER_0"]

    # Agglomerative clustering with distance threshold to decide clusters
    # Heuristic: scale threshold by feature variance
    var = np.var(X)
    threshold = max(1.0, 4.0 * np.sqrt(var))
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, linkage='ward')
    labels = clustering.fit_predict(X)
    return [f"SPEAKER_{int(l)}" for l in labels]
