from typing import List

import numpy as np

class NumpyNetwork:
    def __init__(self, n_input: int, n_hidden: int, n_output: int):
        """
        A minimalistic Neural Network, using numpy.
        - One hidden layer: SoftReLU [0, inf]
        - Output layer: sigmoid (0, 1)

        :param int n_input: Size of input vector
        :param int n_hidden: Size of hidden layer
        :param int n_output: Size of output vector
        """
        n_hidden = n_hidden
        self.n_con1 = n_input * n_hidden
        self.n_con2 = n_hidden * n_output
        self.lin = np.random.uniform(-1, 1, (n_hidden, n_input))
        self.output = np.random.uniform(-1, 1, (n_output, n_hidden))

    def set_weights(self, weights: np.array):
        """
        Set weights of NN.

        :param np.array weights: Vector of weights
        """
        assert len(weights) == self.n_con1 + self.n_con2, f"Got {len(weights)} but expected {self.n_con1 + self.n_con2}"
        weight_matrix1 = weights[:self.n_con1].reshape(self.lin.shape)
        weight_matrix2 = weights[-self.n_con2:].reshape(self.output.shape)
        self.lin = weight_matrix1
        self.output = weight_matrix2

    def forward(self, state: np.array):
        hid_l = np.tanh(np.dot(self.lin, state))
        output_l = np.tanh(np.dot(self.output, hid_l))
        return output_l

class NNController():
    def __init__(self, n_states, n_actions):
        self.controller_type = "NN"
        self.n_input = n_states
        self.n_output = n_actions
        self.model = NumpyNetwork(n_states, n_states, n_actions)
        self.n_params = self.model.n_con1 + self.model.n_con2

    def geno2pheno(self, genotype: np.array):
        self.model.set_weights(genotype)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Given a state, give an appropriate action

        :param <np.array> state: A single observation of the current state, dimension is (state_dim)
        :return: np.ndarray action: A vector of motor inputs
        """

        print("#n_input :", self.n_input)
        print("#state.shape :", state.shape[0])

        assert (state.shape[0] == self.n_input), "State does not correspond with expected input size"
        action = self.model.forward(state)
        return action

