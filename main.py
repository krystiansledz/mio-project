import ga
from BeeHive import *

train_data_input = np.empty((0, 360))
train_data_output = np.empty(0, dtype=int)
test_data_input = np.empty((0, 360))
test_data_output = np.empty(0, dtype=int)

fruits = ["apple", "raspberry", "mango", "lemon"]

train_data_ratio = 0.7

for fruit in fruits:
    file_name = 'data/' + fruit + "_dataset.npy"
    data = np.load(file_name)
    train_data_length = int(np.round(len(data) * train_data_ratio))
    train, test = data[:train_data_length, ...], data[train_data_length:, ...]
    train_data_input = np.append(train_data_input, train, axis=0)
    test_data_input = np.append(test_data_input, test, axis=0)

for fruit in fruits:
    file_name = 'data/' + fruit + "_outputs.npy"
    data = np.load(file_name)
    train_data_length = int(np.round(len(data) * train_data_ratio))
    train, test = data[:train_data_length, ...], data[train_data_length:, ...]
    train_data_output = np.append(train_data_output, train, axis=0)
    test_data_output = np.append(test_data_output, test, axis=0)

print(train_data_input.shape)
print(test_data_input.shape)
print(train_data_output.shape)
print(test_data_output.shape)

with open('test.npy', 'wb') as f:
    np.save(f, np.array([1, 2]))
    np.save(f, np.array([1, 3]))

sol_per_pop = 2
num_parents_mating = 4
num_generations = 1000
mutation_percent = 10

# Creating the initial population.
initial_pop_weights = []
for curr_sol in np.arange(0, sol_per_pop):
    HL1_neurons = 150
    input_HL1_weights = np.random.uniform(low=-0.1, high=0.1,
                                          size=(train_data_input.shape[1],
                                                HL1_neurons))
    HL2_neurons = 60
    HL1_HL2_weights = np.random.uniform(low=-0.1, high=0.1, size=(HL1_neurons,
                                                                  HL2_neurons))
    output_neurons = 4
    HL2_output_weights = np.random.uniform(low=-0.1, high=0.1,
                                           size=(HL2_neurons, output_neurons))
    initial_pop_weights.append(np.array([input_HL1_weights, HL1_HL2_weights,
                                         HL2_output_weights]))

pop_weights_mat = np.array(initial_pop_weights)
print('pop_weights_mat', pop_weights_mat.shape)

pop_weights_vector = ga.mat_to_vector(pop_weights_mat)

hive = BeeHive(lower = -5, upper = 5,
            shape        = [train_data_input.shape[1],150,60,4],
            fitness      = ga.fitness, 
            numb_bees    = 10   ,
            max_itrs     = 100   ,
            max_trials   = 30  ,
            mutation     = 20,
            verbose      = True ,
            input_data   = train_data_input, 
            output_data  = train_data_output,
            input_test_data   = test_data_input, 
            output_test_data  = test_data_output)

hive.run()
