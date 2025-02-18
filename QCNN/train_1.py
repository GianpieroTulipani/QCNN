from QCNN_architecture import QCNN
from pathlib import Path
from loguru import logger
from dataset import load_data_and_process
import numpy as np
import os
import json
import typer
import matplotlib.pyplot as plt
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from config import RAW_DATA_DIR, MODELS_DIR
from tqdm import tqdm

app = typer.Typer()
algorithm_globals.random_seed = 12345

# Define maximum iterations (should match the optimizer's maxiter)
max_iter = 200  
# Create a tqdm progress bar.
progress_bar = tqdm(total=max_iter, desc="Training progress", ncols=80)

# Define a callback that updates the progress bar.
def callback_progress(weights, obj_func_eval):
    progress_bar.update(1)

@app.command()
def train(
    data_dir: Path = RAW_DATA_DIR / 'Brain_Data_Organised',
    weights_dir: Path = MODELS_DIR / "qcnn_initial_point.json",
):
    # Build the QCNN: QCNN() should return (ansatz, qnn)
    ansatz, qnn = QCNN()
    
    if os.path.exists(weights_dir):
        with open(weights_dir, "r") as f:
            initial_point = json.load(f)
        logger.info("Loaded initial weights from JSON.")
    else:
        num_weights = len(ansatz.parameters)
        initial_point = np.random.rand(num_weights).tolist()
        with open(weights_dir, "w") as f:
            json.dump(initial_point, f)
        logger.info("Created new initial weights and saved to JSON.")

    # Use the callback_progress instead of a plotting callback.
    classifier = NeuralNetworkClassifier(
        qnn,
        optimizer=COBYLA(maxiter=max_iter),
        callback=callback_progress,
        initial_point=initial_point,
    )

    X_train, X_test, y_train, y_test = load_data_and_process(data_dir)
    logger.info(f"Shape of training data: {X_train.shape}")

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    classifier.fit(X_train, y_train)
    
    # Close the progress bar when done.
    progress_bar.close()
    
    logger.info(f"Accuracy on training data: {np.round(100 * classifier.score(X_train, y_train), 2)}%")

if __name__ == "__main__":
    app()
