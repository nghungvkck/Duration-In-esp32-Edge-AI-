"""
STATS
=====
Extract statistical features from audio segment.
"""
import sys
from pathlib import Path
import numpy as np
import librosa

_processing_dir = Path(__file__).parent.parent.parent
if str(_processing_dir) not in sys.path:
    sys.path.insert(0, str(_processing_dir))

from config import STATS_FRAME_LENGTH, STATS_HOP_LENGTH


class StatsExtractor:
    """Extract statistical features."""
    
    def __init__(self, frame_length=STATS_FRAME_LENGTH,
                 hop_length=STATS_HOP_LENGTH):
        self.frame_length = frame_length
        self.hop_length = hop_length
    
    def extract(self, segment_data, sr):
        """Extract stats from audio segment."""
        rms = librosa.feature.rms(
            y=segment_data,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )[0]
        
        zcr = librosa.feature.zero_crossing_rate(
            segment_data,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )[0]
        
        centroid = librosa.feature.spectral_centroid(y=segment_data, sr=sr)[0]
        
        return {
            'rms': {
                'mean': float(np.mean(rms)),
                'std': float(np.std(rms)),
                'max': float(np.max(rms)),
            },
            'zcr': {
                'mean': float(np.mean(zcr)),
                'std': float(np.std(zcr)),
            },
            'spectral_centroid': {
                'mean': float(np.mean(centroid)),
            },
            'energy': float(np.sum(segment_data ** 2)),
            'peak_amplitude': float(np.max(np.abs(segment_data))),
            'duration': float(len(segment_data) / sr),
        }