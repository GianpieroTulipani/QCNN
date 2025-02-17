from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from PIL import Image
from loguru import logger
from tqdm import tqdm

from QCNN.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

def load_data_and_process(data_dir: Path, target_size=(256, 256)):
    logger.info(f"Loading data from {data_dir}")
    data = []
    labels = []
    
    subfolders = [subfolder for subfolder in data_dir.iterdir() if subfolder.is_dir()]
    for subfolder in subfolders:
        files = list(subfolder.iterdir())
        for file in tqdm(files, desc="Loading images", total=len(files)):
            with Image.open(file) as img:
                img = img.resize(target_size)
                img_array = np.array(img) / 255.0
                data.append(img_array)
            labels.append(0 if subfolder.name == "Normal" else 1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    
    X_train_flat = np.array([img.flatten() for img in X_train])
    X_test_flat = np.array([img.flatten() for img in X_test])
    
    pca = PCA(n_components=8, random_state=42)
    X_train_pca = pca.fit_transform(X_train_flat)
    X_test_pca = pca.transform(X_test_flat)
    
    return X_train_pca, X_test_pca, y_train, y_test


if __name__ == "__main__":
    data_dir = RAW_DATA_DIR / 'Brain_Data_Organised'
    X_train, X_test, Y_train, Y_test = load_data_and_process(data_dir)
    logger.info(f"X_train shape: {X_train.shape}, Y_train shape: {len(Y_train)}")
    logger.info(f"X_test shape: {X_test.shape}, Y_test shape: {len(Y_test)}")