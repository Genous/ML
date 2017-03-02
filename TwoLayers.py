import numpy as np
import random
import math

# inputs[0] is always 1 (bias term). inputs[1], inputs[2] are actual binary inputs


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def binary(x):
    return 1 if x >= 0.5 else 0

# this function sets inputs[1] & inputs[2] to a random 0 or 1. inputs[0] is always 1 (bias)
def set_inputs(inputs):
    inputs[1] = random.randint(0, 1)
    inputs[2] = random.randint(0, 1)
    return inputs


# returns dot product of inputs . weights vectors
def get_weighted_sum(inputs, weights):
    return np.dot(inputs, weights)


def get_perceptron_output(inputs, weights):
    return sigmoid(get_weighted_sum(inputs, weights))


def get_network_output(inputs, w1_first_layer, w2_first_layer, w_second_layer):
    h[0] = 1
    h[1] = get_perceptron_output(inputs, w1_first_layer)
    h[2] = get_perceptron_output(inputs, w2_first_layer)
    return get_perceptron_output(h, w_second_layer)


def get_xor_gate_output(inputs):
    return 1 if inputs[1] != inputs[2] else 0

# inputs of the first layer
inputs = np.array([1,0,0])

# inputs of the second layer. Outputs of the first layer
h = np.array([1,0,0])

w1_first_layer = np.array([0.0,0.0,0.0])
w2_first_layer = np.array([0.0,0.0,0.0])
w_second_layer = np.array([0.0,0.0,0.0])

rate = 0.3

print("training...")
for numberOfIterations in range(0, 9999):
    set_inputs(inputs)

    h[0] = 1
    h[1] = get_perceptron_output(inputs, w1_first_layer)
    h[2] = get_perceptron_output(inputs, w2_first_layer)
    actual_final_output = get_network_output(inputs, w1_first_layer, w2_first_layer, w_second_layer)

    expectedOutput = get_xor_gate_output(inputs)

    w_second_layer += rate * (expectedOutput - actual_final_output) * actual_final_output * (1 - actual_final_output) * h
    w1_first_layer += rate * (expectedOutput - actual_final_output) * (1 - actual_final_output) * actual_final_output * w_second_layer[1] * (1 - h[1]) * h[1] * inputs
    w2_first_layer += rate * (expectedOutput - actual_final_output) * (1 - actual_final_output) * actual_final_output * w_second_layer[2] * (1 - h[2]) * h[2] * inputs

print("Done")
print(w1_first_layer, w2_first_layer, w_second_layer)
while 1:
    input()
    set_inputs(inputs)
    print(inputs[1:], get_network_output(inputs, w1_first_layer, w2_first_layer, w_second_layer))
