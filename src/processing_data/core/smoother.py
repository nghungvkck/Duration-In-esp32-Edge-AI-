"""
SMOOTHER
========
Denoise and smooth audio signal before peak detection.

Pipeline:
    1. Take absolute value (envelope)
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
        self.gaussian_sigma = gaussian_sigma          # đơn vị: GIÂY
        self.smooth_window_sec = smooth_window_sec
        self.polyorder = polyorder

    # ------------------------------------------------------------------ #
    # Step 1 — Envelope
    # ------------------------------------------------------------------ #
    def take_abs(self, data):
        """
        Lấy biên độ (envelope) của waveform.

        Sóng âm là dao động xoay chiều — đỉnh dương và đỉnh âm
        đều là cùng một sự kiện âm thanh. Dùng np.abs() để đo
        đúng độ lớn tại mỗi thời điểm, không làm mất nửa tín hiệu.
        """
        return np.abs(data)

    # ------------------------------------------------------------------ #
    # Step 2 — Gaussian denoise
    # ------------------------------------------------------------------ #
    def denoise(self, data, sr):
        """
        Giảm nhiễu bằng Gaussian filter.

        SMOOTH_GAUSSIAN_SIGMA tính bằng GIÂY, quy đổi ra số mẫu
        bằng cách nhân với sample rate.
        """
        sigma_samples = self.gaussian_sigma * sr
        if sigma_samples < 0.1:
            # sigma quá nhỏ -> bỏ qua cho nhanh
            return data
        return gaussian_filter(data, sigma=sigma_samples)

    # ------------------------------------------------------------------ #
    # Step 3 — Savitzky-Golay smooth
    # ------------------------------------------------------------------ #
    def smooth(self, data, sr):
        """Smooth signal with Savitzky-Golay filter."""
        window_size = int(sr * self.smooth_window_sec)

        # window phải là số lẻ và >= 3
        if window_size < 3:
            window_size = 3
        if window_size % 2 == 0:
            window_size += 1

        # polyorder bắt buộc < window_size
        poly = min(self.polyorder, window_size - 1)

        return savgol_filter(data, window_size, polyorder=poly)

    # ------------------------------------------------------------------ #
    # Full pipeline
    # ------------------------------------------------------------------ #
    def process(self, data, sr):
        """Full pipeline: abs -> denoise -> smooth."""
        envelope = self.take_abs(data)
        denoised = self.denoise(envelope, sr)
        smoothed = self.smooth(denoised, sr)
        return smoothed


# ---------------------------------------------------------------------- #
# Quick test
# ---------------------------------------------------------------------- #
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

        # Chuẩn hóa raw về [-1, 1] trước khi xử lý
        data = data / np.max(np.abs(data))

        print(f"File: {Path(files[0]).name}")
        print(f"Duration: {len(data)/sr:.2f}s\n")

        envelope = smoother.take_abs(data)
        print(f"Step 1 - abs:      max={envelope.max():.4f}")

        denoised = smoother.denoise(envelope, sr)
        print(f"Step 2 - denoised: max={denoised.max():.4f}")

        smoothed = smoother.smooth(denoised, sr)
        print(f"Step 3 - smoothed: max={smoothed.max():.4f}")

        result = smoother.process(data, sr)
        print(f"\nFull pipeline:     max={result.max():.4f}")