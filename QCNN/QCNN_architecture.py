from QCNN.Unitary import F1, F2, P
import numpy as np
from qiskit import QuantumCircuit


def conv_layer(qcnn, params, F2_params, num_layer, qubits):
    start_index = 0
    n = len(qubits)
    end_index = n * 2
    for l in range(num_layer):
        if n == 8:
            step = n // 2
            for i in range(0, n, step):
                sub_params = params[start_index:end_index]
                sub_qc = F1(sub_params, step)
                qcnn.append(sub_qc.to_instruction(), qubits[i:i+step])
        else:
            end_index += n * 2
            sub_params = params[start_index:end_index]
            sub_qc = F1(sub_params, n)
            qcnn.append(sub_qc.to_instruction(), qubits)

        start_index = end_index
        end_index += F2_params

        for i in range(0, n, 2):
            sub_params = params[start_index:end_index]
            sub_qc = F2(sub_params)
            qcnn.append(sub_qc.to_instruction(), [qubits[i % n], qubits[(i+1) % n]])
        for i in range(1, n, 2):
            sub_params = params[start_index:end_index]
            sub_qc = F2(sub_params)
            qcnn.append(sub_qc.to_instruction(), [qubits[i % n], qubits[(i+1) % n]])

        start_index = end_index
        end_index += n * 2

def pooling_layer1(qcnn, params):
    sub_qc = P(params)
    qcnn.append(sub_qc.to_instruction(), [7, 6])
    qcnn.append(sub_qc.to_instruction(), [1, 0])

def pooling_layer2(qcnn, params):
    sub_qc = P(params)
    qcnn.append(sub_qc.to_instruction(), [3, 2])
    qcnn.append(sub_qc.to_instruction(), [5, 4])

def pooling_layer3(qcnn, params, num_classes):
    if num_classes == 4:
        qcnn.append(P(params).to_instruction(), [2, 0])
    qcnn.append(P(params).to_instruction(), [6, 4])

def QCNN_architecture(qcnn, params, F2_params, num_classes, num_layer, n_qubit):
    index = (F2_params + n_qubit * 2) * num_layer
    param1CL = params[0:index]
    param1PL = params[index:index + 2]

    index1 = (F2_params + (n_qubit - 2) * 4) * num_layer
    param2CL = params[index + 2:index + 2 + index1]
    param2PL = params[index + 2 + index1:index + 2 + index1 + 2]

    param3CL = params[index + 2 + index1 + 2 : index + 2 + index1 + 2 + index]

    conv_layer(qcnn, param1CL, F2_params, num_layer, list(range(n_qubit)))
    pooling_layer1(qcnn, param1PL)

    qcnn.barrier()

    conv_layer(qcnn, param2CL, F2_params, num_layer, [0, 2, 3, 4, 5, 6])
    pooling_layer2(qcnn, param2PL)

    qcnn.barrier()

    conv_layer(qcnn, param3CL, F2_params, num_layer, [0, 2, 4, 6])
    if num_classes in [4, 6, 8]:
        param3PL = params[index + 2 + index1 + 2 + index : index1 + 2 + index1 + 2 + index + 2]
        pooling_layer3(qcnn, param3PL, num_classes)


def QCNN(X, params, F2_params, cost_fn='cross_entropy', num_classes=10, num_layer=1):
    n_qubit = 8
    qcnn = QuantumCircuit(n_qubit, n_qubit)

    # Encoding input features
    for i in range(n_qubit):
        qcnn.h(i)
        qcnn.ry(X[i], i)
    qcnn.barrier()

    # Build the QCNN architecture
    QCNN_architecture(qcnn, params, F2_params, num_classes, num_layer, n_qubit)

    if num_classes == 4:
        measure_wires = [0, 4]
    elif num_classes == 6:
        measure_wires = [0, 2, 4]
    elif num_classes == 8:
        measure_wires = [0, 2, 4]
    else:
        measure_wires = [0, 2, 4, 6]

    for wire in measure_wires:
        qcnn.measure(wire, wire)

    return qcnn

if __name__ == '__main__':
  n_qubit = 8
  X = np.random.random(n_qubit)
  U_params = 15
  num_layer = 1

  total1 = (U_params + n_qubit * 2) * num_layer
  total2 = (U_params + (n_qubit - 2) * 4) * num_layer
  total_params = int(total1 + 2 + total2 + 2 + total1)

  params = np.random.random(total_params)

  qc = QCNN(X, params, U_params, num_classes=10, num_layer=num_layer)