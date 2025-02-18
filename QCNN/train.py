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

app = typer.Typer()
algorithm_globals.random_seed = 12345

objective_func_vals = []

def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective Function Value vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function Value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()

@app.command()
def train(
    data_dir: Path = RAW_DATA_DIR / 'Brain_Data_Organised',
    weights_dir: Path = MODELS_DIR / "qcnn_initial_point.json",
):
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

	classifier = NeuralNetworkClassifier(
		qnn,
		optimizer=COBYLA(maxiter=200),
		callback=callback_graph,
		initial_point=initial_point,
	)

	X_train, X_test, y_train, y_test = load_data_and_process(data_dir)

	logger.info("Shape of training data:", X_train.shape)

	X_train = np.asarray(X_train)
	y_train = np.asarray(y_train)

	classifier.fit(X_train, y_train)
	logger.info(f"Accuracy on training data: {np.round(100 * classifier.score(X_train, y_train), 2)}%")
if __name__ == "__main__":
    app()