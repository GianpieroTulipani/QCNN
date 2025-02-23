from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import tensorflow as tf
from PIL import Image
from loguru import logger
from tqdm import tqdm

def data_load_and_process(num_classes, all_samples, seed):
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0

    if not all_samples:
        num_examples_per_class = 250
        selected_indices = []

        for class_label in range(10):
            indices = np.where(y_train == class_label)[0][:num_examples_per_class]
            selected_indices.extend(indices)

        x_train_subset = x_train[selected_indices]
        y_train_subset = y_train[selected_indices]

        shuffle_indices = np.random.permutation(len(x_train_subset))
        x_train = x_train_subset[shuffle_indices]
        y_train = y_train_subset[shuffle_indices]

    logger.info("Shape of subset training data: {}", x_train.shape)
    logger.info("Shape of subset training labels: {}", y_train.shape)

    mask_train = np.isin(y_train, range(0, num_classes))
    mask_test = np.isin(y_test, range(0, num_classes))

    X_train = x_train[mask_train]
    X_test = x_test[mask_test]
    Y_train = y_train[mask_train]
    Y_test = y_test[mask_test]

    logger.info("Shape of subset training data: {}", X_train.shape)
    logger.info("Shape of subset training labels: {}", Y_train.shape)

    X_train = tf.image.resize(X_train, (784, 1)).numpy()
    X_test = tf.image.resize(X_test, (784, 1)).numpy()
    X_train, X_test = tf.squeeze(X_train), tf.squeeze(X_test)

    pca = PCA(8)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    X_train = (X_train - X_train.min()) * (np.pi / (X_train.max() - X_train.min()))
    X_test = (X_test - X_test.min()) * (np.pi / (X_test.max() - X_test.min()))
    return X_train, X_test, Y_train, Y_test

"""def load_data_and_process(data_dir: Path, target_size=(256, 256)):
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
    
    return X_train_pca, X_test_pca, y_train, y_test"""