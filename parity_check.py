from qiskit import IBMQ
from qiskit.aqua import QuantumInstance
from qiskit.providers.aer import QasmSimulator
from qiskit.aqua.components.optimizers import COBYLA, SPSA
from qiskit.aqua.algorithms import QSVM, VQC
from qiskit import QuantumCircuit
from qiskit.aqua.components import variational_forms
import numpy as np
import itertools
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from utils.var_utils import MyRYRZ
from utils.TEvqc import MyVQC
from utils.quantum_utils import CustomFeatureMap
from utils.bc_utils_ver2 import get_save_model_callback

from qiskit.aqua import set_qiskit_aqua_logging
import logging
set_qiskit_aqua_logging(logging.DEBUG)  # choose INFO, DEBUG to see the log

def run_exp(
    method='naive',
    epochs=200,
    bit=3,
    dup=1,
    reg=0.,
    depth=4,
    seed=111,
    real_device=False
):
    assert bit % 3 == 0, f"number of bit should be x3"
    assert method in ['naive', 'qrac',
                      'te'], f"Method {method} does not exist"
    num_bits = bit

    x_train = []
    y_train = []

    for comb in itertools.product('01', repeat=num_bits):
        comb = [int(x) for x in comb]
        x_train.append(comb)
        y_train.append(sum(comb) % 2)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # Encode information
    if method in ['qrac', 'te']:
        num_qubit = num_bits // 3 * dup
        x_st = []
        for x in x_train:
            x_st.append(''.join(x.astype(str)) * dup)

        x_st = np.array(x_st)
    # Naive
    else:
        num_qubit = num_bits
        x_st = []
        for x in x_train:
            x_st.append(''.join(x.astype(str)) * dup)

        x_st = np.array(x_st)

    vqc_ordinal_log = []

    def loss_history_callback(step, model_params, loss, _, *args):
        vqc_ordinal_log.append(loss)

        # Save model
        temp_model_filename = os.path.join(model_directory, f'step{step}.npz')
        np.savez(temp_model_filename, opt_params = model_params)

    if method == 'naive':
        feature_map = CustomFeatureMap('X', 1, num_qubit)
    elif method == 'te':
        feature_map = QuantumCircuit(num_qubit)
    else:
        feature_map = CustomFeatureMap('ALL3in1', 1, num_qubit)

    if method == 'te':
        var_form = MyRYRZ(num_qubit, depth=depth)
    else:
        var_form = variational_forms.RYRZ(num_qubit, depth=depth)

    training_input = {
        0: x_st[y_train == 0],
        1: x_st[y_train == 1]
    }

    if reg == 0.:
        model_directory = f'models/Parity_check_{method}_{bit}_{dup}_{seed}_{depth}'
    else:
        model_directory = f'models/Parity_check_{method}_{bit}_{dup}_{seed}_{depth}_{reg}'

    if not os.path.isdir(model_directory):
        os.makedirs(model_directory) 

    if method == 'te':
        qsvm = MyVQC(SPSA(epochs), feature_map, var_form, training_input,
                   callback=loss_history_callback, lamb=reg)
    else:
        qsvm = VQC(SPSA(epochs), feature_map, var_form,
                   training_input, callback=loss_history_callback)

    qsvm.random.seed(seed)

    if real_device:
        # Please fix these line...
        provider = IBMQ.get_provider()
        backend = provider.get_backend('ibmq_london')
    else:
        backend = QasmSimulator({"method": "statevector_gpu"})

    quantum_instance = QuantumInstance(
        backend, shots=1024, seed_simulator=seed, seed_transpiler=seed, optimization_level=3)

    result = qsvm.run(quantum_instance)

    y_pred_train = qsvm.predict(x_st)[1]

    # F1 score
    acc = np.mean(y_pred_train == y_train)
    import pickle
    try:
        os.mkdir('models/')
    except:
        pass

    try:
        os.mkdir('results/')
    except:
        pass

    qsvm.save_model(f'{model_directory}/model')
    if reg == 0.:
        with open(f'/content/results/Parity_check_{method}_{bit}_{dup}_{seed}_{depth}', 'wb') as f:
            pickle.dump([vqc_ordinal_log, acc], f)
    else:
        with open(f'/content/results/Parity_check_{method}_{bit}_{dup}_{seed}_{depth}_{reg}', 'wb') as f:
            pickle.dump([vqc_ordinal_log, acc], f)

    print("=" * 97)
    print(f"Method: {method} depth={depth} dup={dup} (reg: {reg})")
    print(f"Number of bit {bit}")
    print(f"Classified: {acc*100}%")
    print("=" * 97)

    return {'train_acc': acc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Choose METHOD')
    parser.add_argument('--method', dest='method',
                        type=str, default='qrac')
    parser.add_argument('--epochs', dest="epochs", type=int, default=1, help="Number of epochs (default 200)")
    parser.add_argument('--dup', dest='dup', type=int, default=1, help="Number of qubit duplication (default 1)")
    parser.add_argument('--bit', dest='bit', type=int, default=3, help="Size of the problem (must be multiplication of 3) (default 3)")
    parser.add_argument('--seed', dest='seed', type=int, default=111)
    parser.add_argument('--reg', dest='reg', type=float, default=0., help="Regularization weight (default 0)")
    parser.add_argument('--depth', dest='depth', type=int, default=4, help="Depth of RYRZ variational form (default 4)")
    parser.add_argument('--real_device', dest='real_device', action='store_true', default=False)
    args = parser.parse_args()
    run_exp(**vars(args))
