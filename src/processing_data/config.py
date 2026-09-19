"""
config.py: Configuration for processing_data module.
"""
from pathlib import Path


# ============================================================
# PATHS
# ============================================================
# Project root (Duration_edge_ai/)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Input
DATA_DIR = PROJECT_ROOT / 'data'
RAW_AUDIO_DIR = DATA_DIR / 'raw_data'

# Output
PROCESSED_DIR = DATA_DIR / 'processed'
MEL_DIR = PROCESSED_DIR / 'mel'
MFCC_DIR = PROCESSED_DIR / 'mfcc'
FFT_DIR = PROCESSED_DIR / 'fft'
SPECTROGRAM_DIR = PROCESSED_DIR / 'spectrograms'


# ============================================================
# AUDIO
# ============================================================
SAMPLE_RATE = 16000
MONO = True
AUDIO_EXTENSIONS = ('.wav', '.mp3', '.m4a', '.flac', '.ogg')


# ============================================================
# SMOOTHER
# ============================================================
SMOOTH_GAUSSIAN_SIGMA = 0.5       # Gaussian filter sigma (denoise)
SMOOTH_WINDOW_SEC = 0.02          # Savitzky-Golay window (seconds)
SMOOTH_POLYORDER = 1              # Savitzky-Golay polyorder


# ============================================================
# PEAK DETECTOR
# ============================================================
PEAK_THRESHOLD_RATIO = 0.15       # threshold = ratio * max(smoothed)
PEAK_MIN_DISTANCE_SEC = 0.3       # min distance between peaks (seconds)


# ============================================================
# SEGMENTER
# ============================================================
SEGMENT_PRE_PEAK_SEC = 0.05       # seconds before peak
SEGMENT_DURATION_SEC = 0.3        # total segment duration (seconds)


# ============================================================
# MEL-SPECTROGRAM
# ============================================================
MEL_N_MELS = 128                  # number of mel bands
MEL_N_FFT = 2048                  # FFT window size
MEL_HOP_LENGTH = 256              # hop length
MEL_FMIN = 50                     # min frequency (Hz)
MEL_FMAX = 2000                   # max frequency (Hz)
MEL_POWER = 2.0                   # power for melspectrogram

# ============================================================
# STATS
# ============================================================
STATS_FRAME_LENGTH = 2048
STATS_HOP_LENGTH = 512