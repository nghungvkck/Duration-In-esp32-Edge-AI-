"""
PIPELINE
========
Main orchestrator.

Vẽ segment đầu tiên dựa trên các tham số trong config.py:
    - SEGMENT_PRE_PEAK_SEC
    - SEGMENT_DURATION_SEC
    - PEAK_THRESHOLD_RATIO
"""
import sys
from pathlib import Path

_processing_dir = Path(__file__).parent
if str(_processing_dir) not in sys.path:
    sys.path.insert(0, str(_processing_dir))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

from core.audio_loader import AudioLoader
from core.smoother import Smoother
from core.peak_detector import PeakDetector
from core.segmenter import Segmenter

from config import (
    PEAK_THRESHOLD_RATIO,
    SEGMENT_PRE_PEAK_SEC,
    SEGMENT_DURATION_SEC,
)


class Pipeline:
    """Main processing pipeline."""

    def __init__(self):
        self.loader = AudioLoader()
        self.smoother = Smoother()
        self.detector = PeakDetector()
        self.segmenter = Segmenter()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize(data):
        """Chuẩn hóa biên độ về [-1, 1]."""
        max_val = np.max(np.abs(data))
        if max_val > 0:
            return data / max_val
        return data

    # ------------------------------------------------------------------ #
    # Main
    # ------------------------------------------------------------------ #
    def analyze_file(self, filepath, show_plot=True):
        """Process a single audio file."""
        print(f"File: {Path(filepath).name}")

        # 1. Load
        data, sr = self.loader.load(filepath)
        print(f"  Loaded: {len(data)/sr:.2f}s @ {sr}Hz")

        # 1b. Normalize raw
        data = self._normalize(data)
        print(f"  Normalized: max(|raw|) = {np.max(np.abs(data)):.3f}")

        # 2. Smooth (envelope)
        smoothed = self.smoother.process(data, sr)
        smoothed = self._normalize(smoothed)

        # 3. Detect peaks
        peaks = self.detector.detect(smoothed, sr)
        print(f"  Peaks: {len(peaks)}")

        # 4. Segment
        segments = self.segmenter.segment_all(data, sr, peaks)
        print(f"  Segments: {len(segments)}")

        # 5. Plot segment đầu tiên
        if show_plot and len(segments) > 0:
            self._plot_first_segment(
                raw=data,
                smoothed=smoothed,
                sr=sr,
                peak_idx=int(peaks[0]),
                title=Path(filepath).name,
            )

        return {
            'filepath': filepath,
            'sample_rate': sr,
            'n_peaks': len(peaks),
            'n_segments': len(segments),
        }

    # ------------------------------------------------------------------ #
    # Plot segment đầu tiên
    # ------------------------------------------------------------------ #
    def _plot_first_segment(self, raw, smoothed, sr, peak_idx, title):
        """
        Vẽ đúng segment đầu tiên theo tham số trong config.py.

        - Segment bắt đầu tại (peak - SEGMENT_PRE_PEAK_SEC)
        - Segment dài đúng SEGMENT_DURATION_SEC
        - Trục x bắt đầu từ 0 (thời gian tương đối trong segment)
        """
        # --- Xác định biên segment theo sample ---
        pre_samples = int(SEGMENT_PRE_PEAK_SEC * sr)
        dur_samples = int(SEGMENT_DURATION_SEC * sr)

        seg_start = max(1, peak_idx - pre_samples)
        seg_end = min(len(raw), seg_start + dur_samples)
        seg_len = seg_end - seg_start

        # --- Cắt dữ liệu ---   
        seg_raw = raw[seg_start:seg_end]
        seg_smooth = smoothed[seg_start:seg_end]

        # --- Trục thời gian tương đối trong segment ---
        t_rel = np.arange(seg_len) / sr
        peak_in_seg = (peak_idx - seg_start) / sr

        # --- Figure ---
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # ================== Panel 1: Waveform segment ================== #
        axes[0].plot(t_rel, seg_raw, color='gray',
                     linewidth=0.7, label='Raw segment')
        axes[0].plot(t_rel, seg_smooth, color='tab:blue',
                     linewidth=1.2, alpha=0.8, label='Envelope')
        axes[0].axvline(peak_in_seg, color='red', linestyle='--',
                        linewidth=1.0, alpha=0.8,
                        label=f'Peak @ {peak_in_seg*1000:.1f}ms')
        axes[0].set_title(
            f'SEGMENT #1 — {title}\n'
            f'[file: {seg_start/sr*1000:.1f} → {seg_end/sr*1000:.1f} ms  |  '
            f'len = {seg_len/sr*1000:.1f} ms]'
        )
        axes[0].set_ylabel('Amplitude')
        y_max = max(np.max(np.abs(seg_raw)), 0.1)
        axes[0].set_ylim(-y_max * 1.1, y_max * 1.1)
        axes[0].grid(True, which='major', alpha=0.3)
        axes[0].legend(loc='upper right', fontsize=8)

        # ================== Panel 2: Envelope segment ================== #
        axes[1].plot(t_rel, np.abs(seg_raw), color='orange',
                     linewidth=0.5, alpha=0.5, label='|Raw|')
        axes[1].plot(t_rel, seg_smooth, color='tab:blue',
                     linewidth=1.4, label='Smoothed')

        # Threshold line
        threshold = PEAK_THRESHOLD_RATIO * smoothed.max()
        axes[1].axhline(threshold, color='green', linestyle='--',
                        linewidth=0.8, alpha=0.7,
                        label=f'Threshold ({PEAK_THRESHOLD_RATIO:.2f})')

        # Peak marker
        peak_val = seg_smooth[int(round(peak_in_seg * sr))]
        axes[1].axvline(peak_in_seg, color='red', linestyle='--',
                        linewidth=1.0, alpha=0.8)
        axes[1].plot(peak_in_seg, peak_val, 'rv', markersize=10,
                     label=f'Peak @ {peak_in_seg*1000:.1f}ms')

        axes[1].set_title(
            f'Envelope of segment — len = {seg_len/sr*1000:.1f} ms '
            f'({seg_len} samples)'
        )
        axes[1].set_xlabel('Time within segment (ms)')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].grid(True, which='major', alpha=0.3)
        axes[1].legend(loc='upper right', fontsize=8)

        # ================== Trục thời gian ================== #
        for ax in axes:
            ax.set_xlim(0, seg_len / sr)
            ax.xaxis.set_major_locator(MultipleLocator(0.02))   # 20 ms
            ax.xaxis.set_minor_locator(MultipleLocator(0.005))  # 5 ms
            ax.tick_params(axis='x', which='major', length=6, labelsize=9)
            ax.tick_params(axis='x', which='minor', length=3)
            ax.grid(True, which='minor', alpha=0.15, linestyle=':')
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f'{x*1000:.0f}')
            )

        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import os

    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)

    pipeline = Pipeline()

    files = pipeline.loader.get_files('data/raw_data/dau_nhua/v1c1')
    if files:
        pipeline.analyze_file(files[0])