import numpy

def sigmoid(inpt):
    return 1.0 / (1 + np.exp(-1 * inpt))

def relu(inpt):
    result = inpt
    result[inpt < 0] = 0
    return result

def update_weights(weights, learning_rate):
    new_weights = weights - learning_rate * weights
    return new_weights

# def train_network(num_iterations, weights, data_inputs, data_outputs, learning_rate, activation="relu"):
#     for iteration in range(num_iterations):
#         print("Itreation ", iteration)
#         for sample_idx in range(data_inputs.shape[0]):
#             r1 = data_inputs[sample_idx, :]
#             for idx in range(len(weights) - 1):
#                 curr_weights = weights[idx]
#                 r1 = np.matmul(r1, curr_weights)
#                 if activation == "relu":
#                     r1 = relu(r1)
#                 elif activation == "sigmoid":
#                     r1 = sigmoid(r1)
#             curr_weights = weights[-1]
#             r1 = np.matmul(r1, curr_weights)
#             predicted_label = np.where(r1 == np.max(r1))[0][0]
#             desired_label = data_outputs[sample_idx]
#             if predicted_label != desired_label:
#                 weights = update_weights(weights,
#                                          learning_rate=0.001)
#     return weights

def predict_outputs(weights, data_inputs, activation="relu"):
    predictions = np.zeros(shape=(data_inputs.shape[0]))
    for sample_idx in range(data_inputs.shape[0]):
        r1 = data_inputs[sample_idx, :]
        for curr_weights in weights:
            r1 = np.matmul(r1, curr_weights)
            if activation == "relu":
                r1 = relu(r1)
            elif activation == "sigmoid":
                r1 = sigmoid(r1)
        predicted_label = np.where(r1 == np.max(r1))[0][0]
        predictions[sample_idx] = predicted_label
    return predictions

f = open("dataset_features.pkl", "rb")
data_inputs2 = pickle.load(f)
f.close()

features_STDs = np.std(a=data_inputs2, axis=0)
data_inputs = data_inputs2[:, features_STDs > 50]

f = open("outputs.pkl", "rb")
data_outputs = pickle.load(f)
f.close()

HL1_neurons = 150
input_HL1_weights = np.random.uniform(low=-0.1, high=0.1,
                                         size=(data_inputs.shape[1], HL1_neurons))
HL2_neurons = 60
HL1_HL2_weights = np.random.uniform(low=-0.1, high=0.1,
                                       size=(HL1_neurons, HL2_neurons))
output_neurons = 4
HL2_output_weights = np.random.uniform(low=-0.1, high=0.1,
                                          size=(HL2_neurons, output_neurons))

weights = np.array([input_HL1_weights,
                       HL1_HL2_weights,
                       HL2_output_weights])

# print('1',len(weights))

# weights = train_network(num_iterations=1,
#                         weights=weights,
#                         data_inputs=data_inputs,
#                         data_outputs=data_outputs,
#                         learning_rate=0.01,
#                         activation="relu")

# print('2', len(weights))

predictions = predict_outputs(weights, data_inputs)
num_flase = np.where(predictions != data_outputs)[0] # acc
print("num_flase ", len(predictions), num_flase.size)
