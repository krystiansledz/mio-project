import numpy as np

def sigmoid(inpt):
    return 1.0 / (1.0 + np.exp(-1 * inpt))

def relu(inpt):
    result = inpt
    result[inpt < 0] = 0
    return result

def predict_outputs(weights_mat, data_inputs, data_outputs, activation="relu"):
    predictions = np.zeros(shape=(data_inputs.shape[0]))
    for sample_idx in range(data_inputs.shape[0]):
        r1 = data_inputs[sample_idx, :]
        for curr_weights in weights_mat:
            r1 = np.matmul(r1, curr_weights)
            if activation == "relu":
                r1 = relu(r1)
            elif activation == "sigmoid":
                r1 = sigmoid(r1)
        predicted_label = np.where(r1 == np.max(r1))[0][0]
        predictions[sample_idx] = predicted_label
    correct_predictions = np.where(predictions == data_outputs)[0].size
    accuracy = (correct_predictions / data_outputs.size) * 100
    return accuracy, predictions

def fitness(weights_mat, data_inputs, data_outputs, activation="relu"):
    accuracy = np.empty(shape=(weights_mat.shape[0]))
    curr_sol_mat = weights_mat[:]
    accuracy, _ = predict_outputs(curr_sol_mat, data_inputs, data_outputs, activation=activation)
    return accuracy