"""
AUDIO LOADER
=============
Load audio files.
"""
import librosa
from pathlib import Path

AUDIO_EXTENSIONS = ('.wav', '.mp3', '.m4a', '.flac', '.ogg')


class AudioLoader:
    """Load audio files."""
    
    def __init__(self, sample_rate=None, mono=True):
        self.sample_rate = sample_rate
        self.mono = mono
    
    def load(self, filepath):
        """Load 1 audio file. Returns (data, sr)."""
        data, sr = librosa.load(filepath, sr=self.sample_rate, mono=self.mono)
        return data, sr
    
    def get_files(self, folder):
        """Get list of audio files in folder (recursive)."""
        folder = Path(folder)
        files = []
        for f in folder.rglob('*'):
            if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS:
                files.append(str(f))
        return sorted(files)


if __name__ == "__main__":
    import os
    project_root = Path(__file__).parent.parent.parent.parent
    os.chdir(project_root)
    
    loader = AudioLoader()
    files = loader.get_files('data/raw_data/dau_nhua/v1c1')
    print(f"Found {len(files)} files")
    
    if files:
        data, sr = loader.load(files[0])
        print(f"Loaded: {Path(files[0]).name}")
        print(f"SR: {sr}")
        print(f"Duration: {len(data)/sr:.2f}s")