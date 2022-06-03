from utils.mnist_utils import *
import numpy as np
import tensorflow_quantum as tfq
import tensorflow as tf
import argparse

THRESHOLD = 0.5

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Rescale the images from [0,255] to the [0.0,1.0] range.
x_train, x_test = x_train[..., np.newaxis] / \
    255.0, x_test[..., np.newaxis]/255.0

print("Number of original training examples:", len(x_train))
print("Number of original test examples:", len(x_test))

x_train, y_train = filter_36(x_train, y_train)
x_test, y_test = filter_36(x_test, y_test)

print("Number of filtered training examples:", len(x_train))
print("Number of filtered test examples:", len(x_test))

x_train_small = tf.image.resize(x_train, (4, 4)).numpy()
x_test_small = tf.image.resize(x_test, (4, 4)).numpy()

# Filter same images
x_train_nocon, y_train_nocon = remove_contradicting(x_train_small, y_train)
x_train_bin = np.array(x_train_nocon > THRESHOLD, dtype=np.int32)
x_test_bin = np.array(x_test_small > THRESHOLD, dtype=np.int32)

def TdEncd(data):
    """
    Encode input data.
    input: [n*n boolean values]

    output: list with len(data) + (3-len(data)) % 3 float numbers
    """
    data = data.reshape(-1).tolist()
    data += [0] * ((3-len(data)) % 3)

    TdEncd = [[ 7.38031422e-01,  3.17597669e+00,  5.57567944e+00], 
            [ 3.36811089e+00,  0.,  1.03425297e-01],
            [-3.80012438e-01,  3.14169982e+00,  4.56768323e+00],
            [ 4.22081847e+00, 0.,  5.62119500e+00],
            [ 5.68533257e+00, -2.50139851e+00,  1.11048557e+00],
            [ 1.18213247e+00,  0.,  2.36461127e+00],
            [ 2.77303818e+00, 0.,  3.95355499e+00],
            [ 1.34769311e+00,  0.,  2.17343336e+00]]

    var_list = []
    for i in range(0, len(data), 2):
        var_list += TdEncd[int("".join(str(j) for j in data[i:i+3]),2)]

    return var_list

data_train, data_test = [], []
for row in x_train_bin:
    data_train.append(TdEncd(row))

for row in x_test_bin:
    data_test.append(TdEncd(row))

data_train = np.array(data_train)
data_test = np.array(data_test)

num_qubit = data_train.shape[1]//3

def convert_to_circuit_TdEncd(data, num_qubit):
    qubits = cirq.GridQubit.rect(num_qubit,1)
    circuit = cirq.Circuit()
    for i in range(num_qubit):
        circuit.append(cirq.rz(data[3*i])(qubits[i]))
        circuit.append(cirq.rx(data[3*i+1])(qubits[i]))
        circuit.append(cirq.rz(data[3*i+2])(qubits[i]))
    return circuit

#Qubits = [cirq.GridQubit(0, i) for i in range(num_qubit)]

x_train_circ = [convert_to_circuit_TdEncd(x, num_qubit) for x in data_train]
x_test_circ = [convert_to_circuit_TdEncd(x, num_qubit) for x in data_test]

x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)

def Network(self, state_transformed_set):
    readout_operators = [cirq.Z(self.Qubits[i]) for i in range(self.n_qubit)]
    init = tf.constant_initializer(self.init_value)
    #init = tf.keras.initializers.RandomUniform(minval=0.0, maxval=0.0)
    #init = tf.keras.initializers.RandomNormal(mean=-0.3, stddev=2)
    inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.int32)   

    # model_appended: circuit(), including encoding and trainable layer
    model_appended = EncodingLayer(tfq.layers.AddCircuit()([self.state_encoding(i) for i in state_transformed_set], append=self.quantum_circuit()))(inputs)
    #model_appended = EncodingLayer(tfq.layers.AddCircuit()(tfq.layers.AddCircuit()([self.state_encoding(i) for i in state_transformed_set], prepend=self.quantum_circuitPre()), append=self.quantum_circuitUlt()))(inputs) #GGH
    if self.noisy_circuit:
        diff = tfq.differentiators.ParameterShift()
        pqc = tfq.layers.Expectation(backend='noisy',differentiator=diff)(model_appended, symbol_names=self.exp_parameter, operators=readout_operators, repetitions=1024, initializer=init)
        #pqc = tfq.layers.SampledExpectation(backend='noisy',differentiator=diff)(model_appended, symbol_names=self.exp_parameter, operators=readout_operators, repetitions=1024, initializer=init)
    else:
        diff = tfq.differentiators.Adjoint()
        pqc = tfq.layers.Expectation(differentiator=diff)(model_appended, symbol_names=self.exp_parameter, operators=readout_operators, initializer=init)
    #network = BiasLayer()(pqc)
    #network = LinearLayer()(pqc)
    #network = Dense(4, kernel_initializer=tf.keras.initializers.Ones(), bias_initializer='zeros')(pqc)
    network = Dense(4, kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.1, maxval=0.1, seed=None), bias_initializer='zeros')(pqc)
    model = tf.keras.Model(inputs=inputs, outputs=network)
    #model = tf.keras.Model(inputs=inputs, outputs=pqc)
    return model