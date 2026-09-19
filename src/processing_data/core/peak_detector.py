"""
PEAK DETECTOR
=============
Detect peaks in smoothed audio signal.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

_processing_dir = Path(__file__).parent.parent
if str(_processing_dir) not in sys.path:
    sys.path.insert(0, str(_processing_dir))

from config import (
    PEAK_THRESHOLD_RATIO,
    PEAK_MIN_DISTANCE_SEC,
)


class PeakDetector:
    """Detect peaks in smoothed audio signal."""
    
    def __init__(self,
                 threshold_ratio=PEAK_THRESHOLD_RATIO,
                 min_distance_sec=PEAK_MIN_DISTANCE_SEC):
        self.threshold_ratio = threshold_ratio
        self.min_distance_sec = min_distance_sec
    
    def detect(self, smoothed_data, sr):
        """Detect peaks in SMOOTHED audio signal."""
        threshold = self.threshold_ratio * np.max(smoothed_data)
        min_distance = int(sr * self.min_distance_sec)
        peaks, _ = find_peaks(smoothed_data, height=threshold, distance=min_distance)
        return peaks


if __name__ == "__main__":
    import os
    project_root = Path(__file__).parent.parent.parent.parent
    os.chdir(project_root)
    
    from core.audio_loader import AudioLoader
    from core.smoother import Smoother
    from core.peak_detector import PeakDetector
    
    loader = AudioLoader()
    smoother = Smoother()
    detector = PeakDetector()
    
    files = loader.get_files('data/raw_data/dau_nhua/v1c1')
    
    if files:
        data, sr = loader.load(files[0])
        smoothed = smoother.process(data, sr)
        peaks = detector.detect(smoothed, sr)
        
        print(f"File: {Path(files[0]).name}")
        print(f"Duration: {len(data)/sr:.2f}s")
        print(f"Peaks: {len(peaks)}")
        print(f"Times: {[f'{p/sr:.2f}s' for p in peaks]}")