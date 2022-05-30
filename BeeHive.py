from Bee import *
import copy
import numpy as np

class BeeHive(object):
    """
    Creates an Artificial Bee Colony (ABC) algorithm.
    The population of the hive is composed of three distinct types
    of individuals:
        1. "employees",
        2. "onlookers",
        3. "scouts".
    The employed bees and onlooker bees exploit the nectar
    sources around the hive - i.e. exploitation phase - while the
    scouts explore the solution domain - i.e. exploration phase.
    The number of nectar sources around the hive is equal to
    the number of actively employed bees and the number of employees
    is equal to the number of onlooker bees.
    """

    
    def __init__(self                 ,
                 lower, upper         ,
                 shape,
                 fitness = None, 
                 numb_bees    =  30   ,
                 max_itrs     = 100   ,
                 max_trials   = None  ,
                 verbose      = False ,
                 input_data   = [], 
                 output_data  = [],
                 pop_weights_mat = []):
        """
        Instantiates a bee hive object.
        1. INITIALISATION PHASE.
        -----------------------
        The initial population of bees should cover the entire search space as
        much as possible by randomizing individuals within the search
        space constrained by the prescribed lower and upper bounds.
        Parameters:
        ----------
            :param list lower          : lower bound of solution vector
            :param list upper          : upper bound of solution vector
            :param int shape            : shape of the solution vector
            :param int numb_bees       : number of active bees within the hive
            :param int max_trials      : max number of trials without any improvment
            :param boolean verbose     : makes computation verbose
        """

        # assigns properties of the optimisation problem
        self.fitness = fitness
        self.lower    = lower
        self.upper    = upper
        self.shape = shape
        self.size = sum([shape[i] * shape[i+1] for i in range(len(shape)-1)])
        print(self.size)

        # computes the number of employees
        self.numb_bees = int((numb_bees + numb_bees % 2))

        # assigns properties of algorithm
    
        self.max_itrs = max_itrs
        
        if (max_trials == None):
            self.max_trials = 0.6 * self.numb_bees * self.size
        else:
            self.max_trials = max_trials

        # initialises current best and its a solution vector
        self.best = 0
        self.solution = None

        # save input_data and output_data
        self.input_data = input_data
        self.output_data = output_data
        
        # creates a bee hive
        self.population = [ Bee(self) for _ in range(self.numb_bees) ]

        # initialises best solution vector to food nectar
        self.find_best()

        # computes selection probability
        self.compute_probability()

        # verbosity of computation
        self.verbose = verbose

    def run(self):
        """ Runs an Artificial Bee Colony (ABC) algorithm. """

        cost = {}; cost["best"] = []; cost["mean"] = []
        for itr in range(self.max_itrs):

            # employees phase
            for index in range(self.numb_bees):
                self.send_employee(index)

            # onlookers phase
            self.send_onlookers()

            # scouts phase
            self.send_scout()

            # computes best path
            self.find_best()

            # stores convergence information
            cost["best"].append( self.best )
            cost["mean"].append( sum( [ bee.value for bee in self.population ] ) / self.numb_bees )

            # prints out information about computation
            if self.verbose:
                self._verbose(itr, cost)

        return cost


    def find_best(self):
        """ Finds current best bee candidate. """

        values = [ bee.value for bee in self.population ]
        index  = values.index(max(values))
        if (values[index] > self.best):
            self.best     = values[index]
            self.solution = self.population[index].vector

    def compute_probability(self):
        """
        Computes the relative chance that a given solution vector is
        chosen by an onlooker bee after the Waggle dance ceremony when
        employed bees are back within the hive.
        """

        # retrieves fitness of bees within the hive
        values = [ bee.value for bee in self.population ]
        max_values = max(values)

        # computes probalities the way Karaboga does in his classic ABC implementation
        self.probas = [0.9 * v / max_values + 0.1 for v in values]
        
        # returns intervals of probabilities
        return [sum(self.probas[:i+1]) for i in range(self.numb_bees)]

    def send_employee(self, index):
        """
        2. SEND EMPLOYED BEES PHASE.
        ---------------------------
        During this 2nd phase, new candidate solutions are produced for
        each employed bee by mutation and mutation of the employees.
        If the modified vector of the mutant bee solution is better than
        that of the original bee, the new vector is assigned to the bee.
        """

        # deepcopies current bee solution vector
        employee = copy.deepcopy(self.population[index])

        # draws a dimension to be crossed-over and mutated
        vector_index = random.randint(0, self.size-1)

        # selects another bee
        bee_ix = index
        while (bee_ix == index): bee_ix = random.randint(0, self.numb_bees-1)

        # produces a child based on current bee and bee's friend
        employee.vector[vector_index] = self._mutation(vector_index, index, bee_ix)

        # computes fitness of child
        employee._fitness()

        # deterministic crowding
        if (employee.value > self.population[index].value):
            self.population[index] = copy.deepcopy(employee)
            self.population[index].counter = 0
        else:
            self.population[index].counter += 1

    def send_onlookers(self):
        """
        3. SEND ONLOOKERS PHASE.
        -----------------------
        We define as many onlooker bees as there are employed bees in
        the hive since onlooker bees will attempt to locally improve the
        solution path of the employed bee they have decided to follow
        after the waggle dance phase.
        If they improve it, they will communicate their findings to the bee
        they initially watched "waggle dancing".
        """

        # sends onlookers
        beta = 0
        for _ in range(self.numb_bees):

            # draws a random number from U[0,1]
            phi = random.random()

            # increments roulette wheel parameter beta
            beta += phi * max(self.probas)
            beta %= max(self.probas)

            # selects a new onlooker based on waggle dance
            index = self.select(beta)

            # sends new onlooker
            self.send_employee(index)


    def select(self, beta):
        """
        4. WAGGLE DANCE PHASE.
        ---------------------
        During this 4th phase, onlooker bees are recruited using a roulette
        wheel selection.
        This phase represents the "waggle dance" of honey bees (i.e. figure-
        eight dance). By performing this dance, successful foragers
        (i.e. "employed" bees) can share, with other members of the
        colony, information about the direction and distance to patches of
        flowers yielding nectar and pollen, to water sources, or to new
        nest-site locations.
        During the recruitment, the bee colony is re-sampled in order to mostly
        keep, within the hive, the solution vector of employed bees that have a
        good fitness as well as a small number of bees with lower fitnesses to
        enforce diversity.
        Parameter(s):
        ------------
            :param float beta : "roulette wheel selection" parameter - i.e. 0 <= beta <= max(probas)
        """

        # computes probability intervals "online" - i.e. re-computed after each onlooker
        self.compute_probability()

        # selects a new potential "onlooker" bee
        for index in range(self.numb_bees):
            if (beta < self.probas[index]):
                return index

    def send_scout(self):
        """
        5. SEND SCOUT BEE PHASE.
        -----------------------
        Identifies bees whose abandonment counts exceed preset trials limit,
        abandons it and creates a new random bee to explore new random area
        of the domain space.
        In real life, after the depletion of a food nectar source, a bee moves
        on to other food sources.
        By this means, the employed bee which cannot improve their solution
        until the abandonment counter reaches the limit of trials becomes a
        scout bee. Therefore, scout bees in ABC algorithm prevent stagnation
        of employed bee population.
        Intuitively, this method provides an easy means to overcome any local
        optima within which a bee may have been trapped.
        """

        # retrieves the number of trials for all bees
        trials = [ self.population[i].counter for i in range(self.numb_bees) ]

        # identifies the bee with the greatest number of trials
        indexes = list(filter(None, [index if t > self.max_trials else None for index, t in enumerate(trials)]))

        # checks if its number of trials exceeds the pre-set maximum number of trials
        for index in indexes:

            # creates a new scout bee randomly
            self.population[index] = Bee(self)

            # sends scout bee to exploit its solution vector
            self.send_employee(index)

    def _mutation(self, dim, current_bee, other_bee):
        """
        Mutates a given solution vector - i.e. for continuous
        real-values.
        Parameters:
        ----------
            :param int dim         : vector's dimension to be mutated
            :param int current_bee : index of current bee
            :param int other_bee   : index of another bee to mutation
        """

        new_value = self.population[current_bee].vector[dim]    + \
               (random.random() - 0.5) * 2                 * \
               (self.population[current_bee].vector[dim] - self.population[other_bee].vector[dim])

        if (new_value < self.lower):
            new_value = self.lower

        if (new_value > self.upper):
            new_value = self.upper

        return new_value

    def _verbose(self, itr, cost):
        """ Displays information about computation. """

        msg = "# Iter = {} | Best Evaluation Value = {} | Mean Evaluation Value = {} "
        print(msg.format(int(itr), cost["best"][itr], cost["mean"][itr]))
