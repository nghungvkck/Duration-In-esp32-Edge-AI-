"""
SMOOTHER
========
Denoise and smooth audio signal before peak detection.

Pipeline:
    1. Take only positive part
    2. Denoise with Gaussian filter
    3. Smooth with Savitzky-Golay filter
"""
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter

_processing_dir = Path(__file__).parent.parent
if str(_processing_dir) not in sys.path:
    sys.path.insert(0, str(_processing_dir))

from config import (
    SMOOTH_GAUSSIAN_SIGMA,
    SMOOTH_WINDOW_SEC,
    SMOOTH_POLYORDER,
)


class Smoother:
    """Denoise and smooth audio signal."""
    
    def __init__(self,
                 gaussian_sigma=SMOOTH_GAUSSIAN_SIGMA,
                 smooth_window_sec=SMOOTH_WINDOW_SEC,
                 polyorder=SMOOTH_POLYORDER):
        self.gaussian_sigma = gaussian_sigma
        self.smooth_window_sec = smooth_window_sec
        self.polyorder = polyorder
    
    def take_positive(self, data):
        """Keep only positive part of the signal."""
        return np.maximum(data, 0)
    
    def denoise(self, data):
        """Reduce noise with Gaussian filter."""
        return gaussian_filter(data, sigma=self.gaussian_sigma)
    
    def smooth(self, data, sr):
        """Smooth signal with Savitzky-Golay filter."""
        window_size = int(sr * self.smooth_window_sec)
        if window_size < 1:
            window_size = 1
        if window_size % 2 == 0:
            window_size += 1
        return savgol_filter(data, window_size, polyorder=self.polyorder)
    
    def process(self, data, sr):
        """Full pipeline: positive -> denoise -> smooth."""
        positive = self.take_positive(data)
        denoised = self.denoise(positive)
        smoothed = self.smooth(denoised, sr)
        return smoothed


if __name__ == "__main__":
    import os
    project_root = Path(__file__).parent.parent.parent.parent
    os.chdir(project_root)
    
    from core.audio_loader import AudioLoader
    
    loader = AudioLoader()
    smoother = Smoother()
    
    files = loader.get_files('data/raw_data/dau_nhua/v1c1')
    
    if files:
        data, sr = loader.load(files[0])
        
        print(f"File: {Path(files[0]).name}")
        print(f"Duration: {len(data)/sr:.2f}s\n")
        
        positive = smoother.take_positive(data)
        print(f"Step 1 - positive: max={positive.max():.4f}")
        
        denoised = smoother.denoise(positive)
        print(f"Step 2 - denoised: max={denoised.max():.4f}")
        
        smoothed = smoother.smooth(denoised, sr)
        print(f"Step 3 - smoothed: max={smoothed.max():.4f}")
        
        result = smoother.process(data, sr)
        print(f"\nFull pipeline: max={result.max():.4f}")