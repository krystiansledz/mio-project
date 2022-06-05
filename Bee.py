import random
import numpy as np


class Bee(object):
    """ Creates a bee object. """

    def __init__(self, bee_hive):
        """
        Instantiates a bee object randomly.
        Parameters:
        ----------
            :param list lower  : lower bound of solution vector
            :param list upper  : upper bound of solution vector
            :param def  fitness    : evaluation function
        """
        self.bee_hive = bee_hive
        self.fitness = bee_hive.fitness
        self.size = bee_hive.size
        self.matrix_shape = np.array([np.zeros((bee_hive.shape[i], bee_hive.shape[i+1])) for i in range(len(bee_hive.shape)-1)])
        # creates a random solution vector
        self._random(bee_hive.lower, bee_hive.upper)

        self._fitness()

        # initialises trial limit counter - i.e. abandonment counter
        self.counter = 0

    def _random(self, lower, upper):
        """ Initialises a solution vector randomly. """
        self.vector = [ lower + random.random() * (upper - lower) for _ in range(self.size)]

    def _fitness(self):
        """ Evaluates the fitness of a solution vector. """
        self.value = self.fitness(self.vector_to_mat(), self.bee_hive.input_data, self.bee_hive.output_data, activation="sigmoid")
        return self.value

    def vector_to_mat(self):
        vector_pop_weights = self.vector
        mat_pop_weights = self.matrix_shape
        mat_weights = []
        
        start = 0
        end = 0
        for layer_idx in range(mat_pop_weights.shape[0]):
            end = end + mat_pop_weights[layer_idx].size
            curr_vector = vector_pop_weights[start:end]
            mat_layer_weights = np.reshape(curr_vector, newshape=(mat_pop_weights[layer_idx].shape))
            mat_weights.append(mat_layer_weights)
            start = end
        return np.reshape(mat_weights, newshape=mat_pop_weights.shape)