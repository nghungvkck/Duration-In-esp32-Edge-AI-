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
    N_MELS
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

if __name__ == "__main__":
    base_path = BASE_PATH
    data_loader = load_Data(base_path)
    data_loader.load_data(base_path)
    