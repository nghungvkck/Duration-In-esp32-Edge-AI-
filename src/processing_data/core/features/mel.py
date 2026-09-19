"""
MEL-SPECTROGRAM
===============
Extract Mel-spectrogram from audio segment.
"""
import sys
from pathlib import Path
import numpy as np
import librosa

# Add processing_data/ to sys.path
_processing_dir = Path(__file__).parent.parent.parent
if str(_processing_dir) not in sys.path:
    sys.path.insert(0, str(_processing_dir))

from config import (
    MEL_N_MELS,
    MEL_N_FFT,
    MEL_HOP_LENGTH,
    MEL_FMIN,
    MEL_FMAX,
    MEL_POWER,
)


class MelExtractor:
    """Extract Mel-spectrogram for CNN."""
    
    def __init__(self,
                 n_mels=MEL_N_MELS,
                 n_fft=MEL_N_FFT,
                 hop_length=MEL_HOP_LENGTH,
                 fmin=MEL_FMIN,
                 fmax=MEL_FMAX,
                 power=MEL_POWER):
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.power = power
    
    def extract(self, segment_data, sr):
        """
        Extract Mel-spectrogram in dB.
        
        Args:
            segment_data: numpy array (audio segment)
            sr: sample rate
        
        Returns:
            mel_db: numpy array (n_mels, time_frames) in dB
        """
        mel_spec = librosa.feature.melspectrogram(
            y=segment_data,
            sr=sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax,
            power=self.power,
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_db