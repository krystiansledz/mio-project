import numpy as np
import ga
import ann
import matplotlib.pyplot
import math
import os
# data_inputs = np.load("dataset_features.npy")

# Optional step of filtering the input data using the standard deviation.
# features_STDs = np.std(a=data_inputs, axis=0)
# data_inputs = data_inputs[:, features_STDs>50]

# data_outputs = np.load("outputs.npy")


# The length of the input vector for each sample (i.e. number of neurons in the input layer).
# num_inputs = data_inputs.shape
# The number of neurons in the output layer (i.e. number of classes).
# num_classes = 4
train_data_input = np.empty((0, 360))
train_data_output =  np.empty((0), dtype=int)
test_data_input = np.empty((0, 360))
test_data_output = np.empty((0), dtype=int)

fruits = ["apple", "raspberry", "mango", "lemon"]

train_data_ratio = 0.9

for fruit in fruits:
    file_name = 'data/' + fruit + "_dataset.npy"
    data = np.load(file_name)
    
    train_data_length = int(np.round(len(data) * train_data_ratio))
    
    train, test = data[:train_data_length, ...],  data[train_data_length:, ...]

    train_data_input = np.append(train_data_input, train, axis=0)
    test_data_input = np.append(test_data_input, test, axis=0)


for fruit in fruits:
    file_name = 'data/' + fruit + "_outputs.npy"
    data = np.load(file_name)

    train_data_length = int(np.round(len(data) * train_data_ratio))

    train, test = data[:train_data_length, ...],  data[train_data_length:, ...]
    
    train_data_output = np.append(train_data_output, train, axis=0)
    test_data_output = np.append(test_data_output, test, axis=0)

    
with open('test.npy', 'wb') as f:
    np.save(f, np.array([1, 2]))
    np.save(f, np.array([1, 3]))
#Genetic algorithm parameters:
#    Mating Pool Size (Number of Parents)
#    Population Size
#    Number of Generations
#    Mutation Percent

sol_per_pop = 2
num_parents_mating = 4
num_generations = 1000
mutation_percent = 10

#Creating the initial population.
initial_pop_weights = []
for curr_sol in np.arange(0, sol_per_pop):
    HL1_neurons = 150
    input_HL1_weights = np.random.uniform(low=-0.1, high=0.1, size=(train_data_input.shape[1], HL1_neurons))

    HL2_neurons = 60
    HL1_HL2_weights = np.random.uniform(low=-0.1, high=0.1, size=(HL1_neurons, HL2_neurons))

    output_neurons = 4
    HL2_output_weights = np.random.uniform(low=-0.1, high=0.1, size=(HL2_neurons, output_neurons))

    initial_pop_weights.append(np.array([input_HL1_weights, HL1_HL2_weights, HL2_output_weights]))

pop_weights_mat = np.array(initial_pop_weights)
pop_weights_vector = ga.mat_to_vector(pop_weights_mat)

print(train_data_input.shape[1])

best_outputs = []
accuracies = np.empty(shape=(num_generations))

# for generation in range(num_generations):
#     print("Generation : ", generation)

#     # converting the solutions from being vectors to matrices.
#     pop_weights_mat = ga.vector_to_mat(pop_weights_vector, 
#                                        pop_weights_mat)

#     # Measuring the fitness of each chromosome in the population.
#     fitness = ann.fitness(pop_weights_mat, 
#                           train_data_input, 
#                           train_data_output, 
#                           activation="sigmoid")

#     accuracies[generation] = fitness[0]
#     print("Fitness")
#     print(fitness)

#     # Selecting the best parents in the population for mating.
#     parents = ga.select_mating_pool(pop_weights_vector, 

#                                     fitness.copy(), 

#                                     num_parents_mating)
#     print("Parents")
#     print(parents)

#     # Generating next generation using crossover.
#     offspring_crossover = ga.crossover(parents,

#                                        offspring_size=(pop_weights_vector.shape[0]-parents.shape[0], pop_weights_vector.shape[1]))

#     print("Crossover")
#     print(offspring_crossover)

#     # Adding some variations to the offsrping using mutation.
#     offspring_mutation = ga.mutation(offspring_crossover, 

#                                      mutation_percent=mutation_percent)
#     print("Mutation")
#     print(offspring_mutation)

#     # Creating the new population based on the parents and offspring.
#     pop_weights_vector[0:parents.shape[0], :] = parents
#     pop_weights_vector[parents.shape[0]:, :] = offspring_mutation

# pop_weights_mat = ga.vector_to_mat(pop_weights_vector, pop_weights_mat)
# best_weights = pop_weights_mat [0, :]
# acc, predictions = ANN.predict_outputs(best_weights, data_inputs, data_outputs, activation="sigmoid")
# print("Accuracy of the best solution is : ", acc)

# matplotlib.pyplot.plot(accuracies, linewidth=5, color="black")
# matplotlib.pyplot.xlabel("Iteration", fontsize=20)
# matplotlib.pyplot.ylabel("Fitness", fontsize=20)
# matplotlib.pyplot.xticks(np.arange(0, num_generations+1, 100), fontsize=15)
# matplotlib.pyplot.yticks(np.arange(0, 101, 5), fontsize=15)

# f = open("weights_"+str(num_generations)+"_iterations_"+str(mutation_percent)+"%_mutation.pkl", "wb")
# pickle.dump(pop_weights_mat, f)
# f.close()