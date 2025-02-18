from Unitary import F1, F2, P

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def encoding_features(qubits):
    num_qubits = len(qubits)
    qc = QuantumCircuit(num_qubits)
    feature_params = ParameterVector('x', num_qubits)
    for i in range(num_qubits):
        qc.h(i)
        qc.ry(feature_params[i], i)
    qc.barrier()
    return qc

def conv_layer(param_prefix, param_length, F2_params, num_layer, qubits):
    num_qubits = len(qubits)
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    params = ParameterVector(param_prefix, length=param_length)
    local_qubits = list(range(num_qubits))

    start_index = 0
    end_index = num_qubits * 2

    for l in range(num_layer):
        if num_qubits == 8:
            step = num_qubits // 2
            for i in range(0, num_qubits, step):
                sub_params = params[start_index:end_index]
                qc.compose(F1(sub_params, step), local_qubits[i:i + step], inplace=True)
        else:
            end_index += num_qubits * 2
            sub_params = params[start_index:end_index]
            qc.compose(F1(sub_params, num_qubits), local_qubits, inplace=True)
        qc.barrier()
        start_index = end_index
        end_index += F2_params

        for i in range(0, num_qubits, 2):
            sub_params = params[start_index:end_index]
            qc.compose(F2(sub_params), [local_qubits[i], local_qubits[(i + 1) % num_qubits]], inplace=True)
        qc.barrier()
        for i in range(1, num_qubits, 2):
            sub_params = params[start_index:end_index]
            qc.compose(F2(sub_params), [local_qubits[i], local_qubits[(i + 1) % num_qubits]], inplace=True)
        start_index = end_index
        end_index += num_qubits * 2

    inst = qc.to_instruction()
    new_qc = QuantumCircuit(num_qubits)
    new_qc.append(inst, new_qc.qubits)
    return new_qc

def pooling_layer(param_prefix, param_length, target_qubits, num_qubits):
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=param_length)
    for tq in target_qubits:
        qc.compose(P(params), [qc.qubits[i] for i in tq], inplace=True)
    inst = qc.to_instruction()
    new_qc = QuantumCircuit(num_qubits)
    new_qc.append(inst, new_qc.qubits)
    return new_qc

def QCNN(n_qubit = 8, num_F2_params=15, num_P_params=2, num_layer=2):
    qubits = list(range(n_qubit))

    feature_map = encoding_features(qubits)

    ansatz = QuantumCircuit(n_qubit, name="Ansatz")

    ansatz.compose(
        conv_layer(
            param_prefix="c1",
            param_length=((n_qubit * 2) + num_F2_params) * num_layer,
            F2_params=num_F2_params,
            num_layer=num_layer,
            qubits=qubits
        ),
        qubits, 
        inplace=True
    )
    ansatz.barrier()

    ansatz.compose(
        pooling_layer(
            param_prefix="p1",
            param_length=num_P_params,
            target_qubits=[[7, 6], [1, 0]],
            num_qubits=n_qubit
        ),
        qubits,
        inplace=True
    )
    ansatz.barrier()

    ansatz.compose(
        conv_layer(
            param_prefix="c2",
            param_length=(((n_qubit - 2) * 4) + num_F2_params) * num_layer,
            F2_params=num_F2_params,
            num_layer=num_layer,
            qubits=[0, 2, 3, 4, 5, 6]
        ),
        [0, 2, 3, 4, 5, 6],
        inplace=True
    )
    ansatz.barrier()

    ansatz.compose(
        pooling_layer(
            param_prefix="p2",
            param_length=num_P_params,
            target_qubits=[[3, 2], [5, 4]],
            num_qubits=n_qubit
        ),
        qubits,
        inplace=True
    )
    ansatz.barrier()

    ansatz.compose(
        conv_layer(
            param_prefix="c3",
            param_length=((4 * 4) + num_F2_params) * num_layer,
            F2_params=num_F2_params,
            num_layer=num_layer,
            qubits=[0, 2, 4, 6]
        ),
        [0, 2, 4, 6],
        inplace=True
    )
    ansatz.barrier()

    ansatz.compose(
        pooling_layer(
            param_prefix="p3",
            param_length=num_P_params,
            target_qubits=[[2, 0], [6, 4]],
            num_qubits=n_qubit
        ),
        qubits,
        inplace=True
    )
    ansatz.barrier()

    ansatz.compose(
        conv_layer(
            param_prefix="c4",
            param_length=((2 * 4) + num_F2_params) * num_layer,
            F2_params=num_F2_params,
            num_layer=num_layer,
            qubits=[0, 4]
        ),
        [0, 4],
        inplace=True
    )
    ansatz.barrier()

    ansatz.compose(
        pooling_layer(
            param_prefix="p4",
            param_length=num_P_params,
            target_qubits=[[4, 0]],
            num_qubits=n_qubit
        ),
        qubits,
        inplace=True
    )

    circuit = QuantumCircuit(n_qubit)
    circuit.compose(feature_map, list(range(n_qubit)), inplace=True)
    circuit.compose(ansatz, list(range(n_qubit)), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * (n_qubit - 1), 1)])
    estimator = Estimator()

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )

    return ansatz, qnn