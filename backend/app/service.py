"""
Pipeline service - wrap processing_data pipeline for API.
"""
import sys
from pathlib import Path
import numpy as np

_project_root = Path(__file__).parent.parent.parent
_processing_dir = _project_root / 'src' / 'processing_data'

if str(_processing_dir) not in sys.path:
    sys.path.insert(0, str(_processing_dir))

from core.audio_loader import AudioLoader
from core.smoother import Smoother
from core.peak_detector import PeakDetector
from core.segmenter import Segmenter
from core.features.mel import MelExtractor
from core.features.stats import StatsExtractor


class PipelineService:
    """Wrap pipeline for API."""
    
    def __init__(self):
        self.loader = AudioLoader()
        self.smoother = Smoother()
        self.detector = PeakDetector()
        self.segmenter = Segmenter()
        self.mel_extractor = MelExtractor()
        self.stats_extractor = StatsExtractor()
    
    # ========================================================
    # SUMMARY: metadata + list segments (NO Mel matrix)
    # ========================================================
    def analyze_summary(self, filepath):
        """
        Analyze file, return summary.
        Does NOT include Mel matrix (too heavy).
        """
        data, sr = self.loader.load(filepath)
        if data is None:
            return {'success': False, 'message': 'Cannot load file'}
        
        smoothed = self.smoother.process(data, sr)
        peaks = self.detector.detect(smoothed, sr)
        segments = self.segmenter.segment_all(data, sr, peaks)
        
        # Summary per segment (stats only, no Mel)
        segments_summary = []
        for i, seg in enumerate(segments):
            seg_data = seg['data']
            stats = self.stats_extractor.extract(seg_data, sr)
            
            segments_summary.append({
                'index': i,
                'peak_time': seg['peak_time'],
                'start_time': seg['start_time'],
                'end_time': seg['end_time'],
                'duration': len(seg_data) / sr,
                'rms_mean': stats['rms']['mean'],
                'zcr_mean': stats['zcr']['mean'],
                'energy': stats['energy'],
                'peak_amplitude': stats['peak_amplitude'],
            })
        
        return {
            'success': True,
            'file_name': Path(filepath).name,
            'sample_rate': sr,
            'duration': len(data) / sr,
            'n_peaks': len(peaks),
            'peaks': [
                {'index': i, 'time': float(p / sr)}
                for i, p in enumerate(peaks)
            ],
            'segments': segments_summary,
            'waveform': self._downsample(data, max_points=2000).tolist(),
        }
    
    # ========================================================
    # SEGMENT DETAIL: full data (WITH Mel matrix)
    # ========================================================
    def analyze_segment(self, filepath, segment_index):
        """
        Get full data for ONE segment (including Mel matrix).
        """
        data, sr = self.loader.load(filepath)
        if data is None:
            return {'success': False, 'message': 'Cannot load file'}
        
        smoothed = self.smoother.process(data, sr)
        peaks = self.detector.detect(smoothed, sr)
        segments = self.segmenter.segment_all(data, sr, peaks)
        
        # Validate index
        if segment_index < 0 or segment_index >= len(segments):
            return {
                'success': False,
                'message': f'Invalid index {segment_index}. '
                           f'File has {len(segments)} segments.',
            }
        
        seg = segments[segment_index]
        seg_data = seg['data']
        
        # Extract Mel + stats
        mel_db = self.mel_extractor.extract(seg_data, sr)
        stats = self.stats_extractor.extract(seg_data, sr)
        
        return {
            'success': True,
            'segment_index': segment_index,
            'peak_time': seg['peak_time'],
            'start_time': seg['start_time'],
            'end_time': seg['end_time'],
            'sample_rate': sr,
            'waveform': seg_data.tolist(),
            'mel_db': mel_db.tolist(),
            'mel_shape': list(mel_db.shape),
            'stats': {
                'rms_mean': stats['rms']['mean'],
                'rms_std': stats['rms']['std'],
                'zcr_mean': stats['zcr']['mean'],
                'spectral_centroid_mean': stats['spectral_centroid']['mean'],
                'energy': stats['energy'],
                'peak_amplitude': stats['peak_amplitude'],
                'duration': stats['duration'],
            },
        }
    
    # ========================================================
    # Helper
    # ========================================================
    @staticmethod
    def _downsample(data, max_points=2000):
        """Downsample array for display."""
        if len(data) <= max_points:
            return data
        step = len(data) // max_points
        indices = np.arange(0, len(data), step)[:max_points]
        return data[indices]