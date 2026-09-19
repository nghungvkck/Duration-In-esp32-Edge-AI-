"""
PIPELINE
========
Main orchestrator.
"""
import sys
from pathlib import Path

_processing_dir = Path(__file__).parent
if str(_processing_dir) not in sys.path:
    sys.path.insert(0, str(_processing_dir))

from core.audio_loader import AudioLoader
from core.smoother import Smoother
from core.peak_detector import PeakDetector
from core.segmenter import Segmenter
from core.features.mel import MelExtractor


class Pipeline:
    """Main processing pipeline."""
    
    def __init__(self):
        self.loader = AudioLoader()
        self.smoother = Smoother()
        self.detector = PeakDetector()
        self.segmenter = Segmenter()
        self.mel_extractor = MelExtractor()
    
    def analyze_file(self, filepath):
        """Process a single audio file."""
        print(f"File: {Path(filepath).name}")
        
        # 1. Load
        data, sr = self.loader.load(filepath)
        print(f"  Loaded: {len(data)/sr:.2f}s @ {sr}Hz")
        
        # 2. Smooth
        smoothed = self.smoother.process(data, sr)
        
        # 3. Detect peaks
        peaks = self.detector.detect(smoothed, sr)
        print(f"  Peaks: {len(peaks)}")
        
        # 4. Segment
        segments = self.segmenter.segment_all(data, sr, peaks)
        print(f"  Segments: {len(segments)}")
        
        # 5. Extract Mel
        for i, seg in enumerate(segments):
            mel_db = self.mel_extractor.extract(seg['data'], sr)
            print(f"    [{i+1}] mel shape={mel_db.shape}")
        
        return {
            'filepath': filepath,
            'sample_rate': sr,
            'n_peaks': len(peaks),
            'n_segments': len(segments),
        }


if __name__ == "__main__":
    import os
    
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    
    pipeline = Pipeline()
    
    # Test 1 file
    files = pipeline.loader.get_files('data/raw_data/dau_nhua/v1c1')
    if files:
        pipeline.analyze_file(files[0])