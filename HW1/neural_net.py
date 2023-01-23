import matplotlib.pyplot as plt

from forward import *
from backward import *


def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size):
    """
    Implements a L-layer neural network.
    All layers but the last should have the ReLU activation function,
    and the final layer will apply the softmax activation function.
    The size of the output layer should be equal to the number of labels in the data.
    Please select a batch size that enables your code to run well (i.e. no memory overflows while still running relatively fast).

    :param X:  the input data, a numpy array of shape (height*width , number_of_examples)
    :param Y:  the “real” labels of the data, a vector of shape (number of examples, 1)
    :param layers_dims: a list containing the dimensions of each layer, including the input
    :param learning_rate:
    :param num_iterations:
    :param batch_size: the number of examples in a single training batch.
    :return:
    parameters – the parameters learnt by the system during the training
                (the same parameters that were updated in the update_parameters function).
    costs – the values of the cost function (calculated by the compute_cost function).
    One value is to be saved after each 100 training iterations (e.g. 3000 iterations -> 30 values)
    """

    def create_batches(input_data, input_labels, batch_size):
        length_data = input_data.shape[-1]
        indices = np.array(list(range(length_data)))
        np.random.shuffle(indices)

        for i in range(0, length_data, batch_size):
            batch_indices = indices[i:i + batch_size]
            yield input_data[:, batch_indices], input_labels[batch_indices]

    epsilon = 1e-2
    costs = []
    valid_accs = []
    iterations_save = []
    parameters = initialize_parameters(layers_dims)

    (x_train, y_train), (x_valid, y_valid) = split_train(X, Y, p=20)  # split with 20 percent validation

    to_watch = 100
    steps = 1
    last_valid_acc = []
    done = False
    train_iterations = 1

    while not done:
        for i in range(1, num_iterations + 1):
            batch_gen = create_batches(x_train, y_train, batch_size)
            curr_iteration_mean_loss = []

            for batch_num, (batch, y) in enumerate(batch_gen):
                AL, caches = L_model_forward(batch, parameters, use_batchnorm=False)
                cost = compute_cost(AL, y)
                grads = L_model_backward(AL, y, caches)
                parameters = Update_parameters(parameters, grads, learning_rate)
                curr_iteration_mean_loss.append(cost)

                if steps % 100 == 0:
                    costs.append(cost)
                    iterations_save.append(steps)
                    valid_acc = Predict(x_valid, y_valid, parameters)
                    valid_accs.append(valid_acc)
                    last_valid_acc.append(valid_acc)

                    is_better = len(last_valid_acc) < to_watch or \
                                max(last_valid_acc) - min(last_valid_acc) > epsilon

                    if not is_better:
                        done = True
                        break
                    else:
                        if len(last_valid_acc) >= to_watch:
                            last_valid_acc.pop(0)

                steps += 1

            avg_loss = np.average(curr_iteration_mean_loss)
            avg_acc = np.average(last_valid_acc)
            print(f"Training iteration: {train_iterations}, Epoch {i} mean loss: {avg_loss:.5f}, "
                  f"Mean validation accuracy: {avg_acc*100:.5f}%\n" + "=" * 80)
            if done:
                break

        train_iterations += 1

    train_acc = Predict(x_train, y_train, parameters)
    print(f'Last training step train accuracy: {train_acc*100:.5f}%')

    valid_acc = Predict(x_valid, y_valid, parameters)
    print(f'Last validation step train accuracy: {valid_acc*100:.5f}%')
    plt.plot(iterations_save, costs)
    plt.ylabel('Loss')
    plt.xlabel('Iteration #')
    plt.grid()
    plt.show()
    plt.plot(iterations_save, valid_accs)
    plt.ylabel('Validation Accuracy')
    plt.xlabel('Iteration #')
    plt.grid()
    plt.show()

    return parameters, costs


def Predict(X, Y, parameters):
    """
    The function receives an input data and the true labels
    and calculates the accuracy of the trained neural network on the data.

    :param X: the input data, a numpy array of shape (height*width, number_of_examples)
    :param Y: the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    :param parameters: a python dictionary containing the DNN architecture’s parameters
    :return:
    accuracy – the accuracy measure of the neural net on the provided data
    (i.e. the percentage of the samples for which the correct label receives the hughest confidence score).
    Use the softmax function to normalize the output values
    """
    y_hat, _ = L_model_forward(X, parameters, use_batchnorm=False)  # y_hat shape: (num_classes, num_examples)
    y_hat_preds = np.argmax(y_hat, axis=0)

    return (y_hat_preds == Y).mean()


def split_train(x_train, y_train, p: float):
    """
    Auxiliary function to split the train into train and validation datasets
    :param x_train: Train data, size [features, examples]
    :param y_train: Train labels, size [examples, 1]
    :param p: Percentage of the validation of all the training data
    :return:
    train_tuple: Train data and labels
    valid_tuple: Validation data and labels
    """
    valid_size = int(x_train.shape[-1] * p / 100)
    indices = np.array(list(range(x_train.shape[-1])))
    np.random.shuffle(indices)
    valid_indices = indices[:valid_size]
    train_indices = indices[valid_size:]
    train_tuple = (x_train[:, train_indices], y_train[train_indices])
    valid_tuple = (x_train[:, valid_indices], y_train[valid_indices])

    return train_tuple, valid_tuple