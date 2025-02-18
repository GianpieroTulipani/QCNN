from qiskit import QuantumCircuit

def F1(params, num_qubits):
    qc = QuantumCircuit(num_qubits)
    n = num_qubits

    for i in range(n):
        qc.ry(params[i], i)

    for i in range(n):
        qc.crx(params[i + n], (i - 1) % n, i)
    qc.barrier()

    for i in range(n):
        qc.ry(params[i + 2*n], i)

    if (n % 3 == 0) or (n == 2):
        for i in reversed(range(n)):
            qc.crx(params[i + 3*n], i, (i - 1) % n)
    else:
        control = n - 1
        target = (control + 3) % n
        for i in reversed(range(n)):
            qc.crx(params[i + 3*n], control, target)
            control = target
            target = (control + 3) % n
    return qc

def F2(params):
    qc = QuantumCircuit(2)
    qc.u(params[0], params[1], params[2], 0)
    qc.u(params[3], params[4], params[5], 1)
    qc.cx(0, 1)
    qc.ry(params[6], 0)
    qc.rz(params[7], 1)
    qc.cx(1, 0)
    qc.ry(params[8], 0)
    qc.cx(0, 1)
    qc.u(params[9], params[10], params[11], 0)
    qc.u(params[12], params[13], params[14], 1)
    return qc

def P(params):
    qc = QuantumCircuit(2)
    qc.crz(params[0], 0, 1)
    qc.x(0)
    qc.crx(params[1], 0, 1)
    return qc