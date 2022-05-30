import random
import ann

class Bee(object):
    """ Creates a bee object. """

    def __init__(self, bee_hive, lower, upper, fun, funcon=None):
        """
        Instantiates a bee object randomly.
        Parameters:
        ----------
            :param list lower  : lower bound of solution vector
            :param list upper  : upper bound of solution vector
            :param def  fun    : evaluation function
            :param def  funcon : constraints function, must return a boolean
        """
        self.bee_hive = bee_hive

        # creates a random solution vector
        self._random(lower, upper)

        self._fitness()

        # initialises trial limit counter - i.e. abandonment counter
        self.counter = 0

    def _random(self, lower, upper):
        """ Initialises a solution vector randomly. """

        self.vector = []
        for i in range(len(lower)):
            self.vector.append( lower[i] + random.random() * (upper[i] - lower[i]) )

    def _fitness(self):
        """
        Evaluates the fitness of a solution vector.
        The fitness is a measure of the quality of a solution.
        """

        self.fitness = ann.fitness(self.vector, self.bee_hive.input_data, self.bee_hive.output_data, activation="sigmoid")