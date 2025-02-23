from QCNN_architecture import QCNN
from pathlib import Path
from loguru import logger
from dataset import data_load_and_process
import numpy as np
import pickle
import typer
import matplotlib.pyplot as plt
from IPython.display import clear_output
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import Sampler
from qiskit_machine_learning.algorithms.classifiers import VQC


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

def interpret(bitstring):
    if isinstance(bitstring, str):
        return int(bitstring, 2) % 4
    elif isinstance(bitstring, (int, float)):
        return int(bitstring) % 4
    else:
        raise TypeError("Invalid input for interpret function")

@app.command()
def train():
    feature_map, ansatz = QCNN()

    sampler = Sampler(options={
        'seed': 0,
        'shots': 4096
    })

    output_shape = 4

    vqc = VQC(
      sampler=sampler,
      feature_map=feature_map,
      ansatz=ansatz,
      optimizer=COBYLA(maxiter=1000),
      callback=callback_graph,
      interpret=interpret,
      loss='cross_entropy',
      output_shape=output_shape
    )

    X_train, _, y_train, _ = data_load_and_process(num_classes=4, all_samples=False, seed=None)

    logger.info("Shape of training data: {}", X_train.shape)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    vqc.fit(X_train, y_train)
    accuracy = np.round(100 * vqc.score(X_train, y_train), 2)
    logger.info(f"Accuracy on training data: {accuracy}%")

    model_filename = "/content/drive/MyDrive/vqc_model.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(vqc, f)
    logger.info(f"Model saved to {model_filename}")
      
if __name__ == "__main__":
    app()