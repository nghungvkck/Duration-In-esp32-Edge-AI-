#=======================================
#file load data và làm giàu dữ liệu qua trình traning
#======================================


import os
from pathlib import Path
import sys
from pathlib import Path
from networkx import config
import numpy as np
from collections import Counter
import numpy as np 
import pandas as pd
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

__ROOT_DIR = Path(__file__).parent.parent
if str(__ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(__ROOT_DIR))

from config.config_model_cnn import (
    BASE_PATH,
    FACTOR_RANGE,
    N_MELS,
    NOISE_FACTOR,
    SHIFT_MAX
)

class load_Data:
    def __init__(self, base_path):
        self.base_path = base_path
        self.label_encoder = LabelEncoder()
        
    def load_data(self, base_path=None):
        X_list = []
        y_list = []
        for label in ['xanh', 'chin']:
            folder_path = os.path.join(self.base_path, label)
            if not os.path.exists(folder_path):
                print(f"Folder {folder_path} does not exist.")
                continue
            
            files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
            for file in files:
                file_path = os.path.join(folder_path, file)
                try:
                    data = np.load(file_path)
                    if data.ndim != 2:
                        continue  # Skip if data is not 2D
                    if data.shape[0] != N_MELS and data.shape[1] != N_MELS:
                        data = data.T  # Transpose if the shape is not (N_MELS, 128):
                    if data.shape[0] == N_MELS:
                        X_list.append(data)
                        y_list.append(label)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        # print(f"Loaded {len(X_list)} samples.")
        # print(f"Class distribution: {Counter(y_list)}")
        
        
        # Convert lists to numpy arrays
        X = np.asarray(X_list, dtype=np.float32)
        y = self.label_encoder.fit_transform(y_list)
        
        # Chuẩn hóa
        X = (X - np.min(X)) / (np.max(X) - np.min(X))
        X= np.expand_dims(X, axis=-1)  # Add channel dimension
        
        print("\n LOAD COMPLETE!")
        print(f"   - Total samples: {len(X)}")
        print(f"   - X shape: {X.shape}")
        print(f"   - y shape: {y.shape}")
        
        return X, y
    
    # Enrichment data augmentation: Resize spectrograms to a target shape 
    def add_noise(self, mel_spec, noise_factor=NOISE_FACTOR):
        # thêm nhiễu gaussian vào spectrogram
        noise = np.random.randn(*mel_spec.shape)* noise_factor
        return mel_spec + noise
    
    def time_shift(self, mel_spec, shift_max=SHIFT_MAX):
        # dịch chuyển thời gian của spectrogram
        shift = np.random.randint(-shift_max, shift_max)
        return np.roll(mel_spec, shift, axis=1)
    
    def adjust_brightness(self, mel_spec, factor_range=FACTOR_RANGE):
        # điều chỉnh độ sáng của spectrogram
        factor = np.random.uniform(*factor_range)
        return mel_spec * factor
    
    def augemnt_chin_data (self, X_chin, target_count= None):
        if target_count is None:
            target_count = len(X_chin) * 5
            
        X_aug = []
        current_count = len(X_chin)
        
        while len(X_aug) < target_count:
            idx = np.random.randint(0, current_count)
            x = X_chin[idx].copy()
            
            # Randomly choose an augmentation method
            aug_type = np.random.choice(['noise', 'shift', 'brightness', 'combined'])
            
            if aug_type == 'noise':
                x = self.add_noise(x, noise_factor = np.random.uniform(0.001, 0.01))
            if aug_type == 'shift':
                x = self.time_shift(x, shift_max = np.random.randint(5,15))
            if aug_type == 'brightness':
                x = self.adjust_brightness(x, factor_range = (0.8, 1.2))
            else:
                x= self.add_noise(x, 0.001)
                if np.random.rand() > 0.5:
                    x = self.time_shift(x, 10)
                if np.random.rand() > 0.5:
                    x = self.adjust_brightness(x)
            X_aug.append(x)
        return np.array(X_aug)
            
    
    def resize_spectrograms(self, X, target_size=(N_MELS, N_MELS)):
        X_resized = []
        for x in X:
            if x.ndim == 3:
                x = x.squeeze(-1)  # Remove channel dimension if present
            if x.ndim == 2:
                if x.shape[0] != target_size[0]:
                    x = x.T 
                x_resized = resize(x, target_size, mode='constant', anti_aliasing=True)
                X_resized.append(x_resized)
            else:
                raise ValueError(f"Unexpected shape {x.shape} for spectrogram.")
        return np.array(X_resized)
                

if __name__ == "__main__":
    base_path = BASE_PATH
    data_loader = load_Data(base_path)
    data_loader.load_data(base_path)
    