import numpy as np
import random
import math

# inputs[0] is always 1 (bias term). inputs[1], inputs[2] are actual binary inputs


def sigmoid(x):
  return 1 / (1 + math.exp(-x))


# this function sets inputs[1] & inputs[2] to a random 0 or 1
def set_inputs(inputs):
    inputs[1] = random.randint(0, 1)
    inputs[2] = random.randint(0, 1)
    return inputs


# returns dot product of inputs . weights vectors
def get_weighted_sum(inputs, weights):
    return np.dot(inputs, weights)


def get_perceptron_output(inputs, weights):
    return sigmoid(get_weighted_sum(inputs, weights))


def get_and_gate_output(inputs):
    return 1 if (inputs[1] == inputs[2] == 1) else 0

inputs = np.array([1,0,0])
weights = np.array([0.0,0.0,0.0])

rate = 2

print("training...")
for numberOfIterations in range(0, 99999):
    set_inputs(inputs)
    actualOutput = get_perceptron_output(inputs, weights)
    expectedOutput = get_and_gate_output(inputs)

    weights += rate * (expectedOutput - actualOutput) * actualOutput * (1 - actualOutput) * inputs

print("Done")
print("Weights:", weights)

while 1:
    input()
    set_inputs(inputs)
    print(inputs[1:], get_perceptron_output(inputs, weights))
