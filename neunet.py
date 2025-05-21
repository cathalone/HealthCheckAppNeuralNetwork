import numpy as np


class NeuralNetwork:
    def __init__(self, neural_network_structure, activation_functions_list,
                 minimum=0, maximum=1, bias=1):
        self.neural_network_ = neural_network_structure
        self.weights_ = self.random_weights(maximum, minimum)
        self.bias_ = bias
        self.activation_functions_list_ = activation_functions_list

    def __str__(self):
        st = ''
        for i in self.neural_network_:
            st += str(i) + '  '
        return st

    def neuron_value(self, value_arr, weight_arr, act_func):
        x = np.dot(value_arr, np.transpose(weight_arr))
        res = act_func(x)
        return res

    def neuron_value_list(self, input_data):
        number_of_iterations = len(self.activation_functions_list_)

        value_list = []

        for i in range(number_of_iterations):
            def f(x):
                if self.activation_functions_list_[i] == 'identity':
                    return x
                elif self.activation_functions_list_[i] == 'sigmoid':
                    return 1 / (1 + np.exp(-x))
                elif self.activation_functions_list_[i] == 'th':
                    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

            if i == 0:
                value = np.append(f(input_data), self.bias_)
                value_list.append(value)
            elif i == number_of_iterations - 1:
                value = self.neuron_value(value, self.weights_[i - 1], f)
                value_list.append(value)
            else:
                value = np.append(self.neuron_value(value, self.weights_[i - 1], f), self.bias_)
                value_list.append(value)

        return value_list

    def backpropagation(self, value_list, expected_results, education_speed):
        weight_list_reversed = self.weights_[::-1]
        value_list_reversed = value_list[::-1]
        activation_functions_list_reversed = self.activation_functions_list_[::-1]

        delta_w_list = []

        for i in range(len(value_list_reversed)):
            def derivative(layer_output):
                if activation_functions_list_reversed[i] == 'identity':
                    return 1
                elif activation_functions_list_reversed[i] == 'sigmoid':
                    return (1 - layer_output) * layer_output
                elif activation_functions_list_reversed[i] == 'th':
                    return 1 - (layer_output * layer_output)

            if i == 0:
                delta = (expected_results - value_list_reversed[i]) * derivative(value_list_reversed[i])
            else:
                gradient = np.matmul(np.transpose(np.array([delta])), np.array([value_list_reversed[i]]))
                delta_w = education_speed * gradient
                delta_w_list.append(delta_w)
                delta = (np.sum(delta * np.transpose(weight_list_reversed[i - 1]), axis=1) * derivative(
                    value_list_reversed[i]))[:-1]

        return delta_w_list[::-1]

    def random_weights(self, minimum, maximum):
        weights_list = []

        for i in range(len(self.neural_network_) - 1):
            weights = np.random.uniform(minimum, maximum, (self.neural_network_[i + 1], self.neural_network_[i] + 1))
            weights_list.append(weights)

        return weights_list

    def train(self, input_data, expected_result, number_of_repetitions=1000, education_speed=0.01, progress_bar=False):
        if progress_bar == False:
            def progress(i, number_of_iterations, number_of_repetitions):
                pass
        else:
            def progress(i, number_of_iterations, number_of_repetitions):
                pr_bar = round((i / number_of_iterations) * 30)
                holder = ' ' * 30 + '\r'
                print(holder + "epoch(" + str(j + 1) + "/" + str(
                    number_of_repetitions) + ")  " + '|' + '■' * pr_bar + '-' * (30 - pr_bar) + '|', end='')

        number_of_iterations = len(input_data)

        for j in range(number_of_repetitions):

            for i in range(number_of_iterations):

                progress(i, number_of_iterations, number_of_repetitions)

                value_list = self.neuron_value_list(input_data[i])
                delta_w = self.backpropagation(value_list, expected_result[i], education_speed)
                for k in range(len(self.weights_)):
                    self.weights_[k] += delta_w[k]

            if progress_bar == False:
                pass
            else:
                print()

    def check(self, input_data):
        result = self.neuron_value_list(input_data)[-1]
        return result


def loss_func(obtained, expected):
    res = 0
    for i in range(len(obtained)):
        res += (obtained[i] - expected[i]) ** 2
    res /= len(obtained)
    return res


def save_weights(neural_network):
    for i in range(len(neural_network.weights_)):
        current = neural_network.weights_[i]
        np.savetxt("weights\\weights" + str(i + 1) + ".txt", current)


def load_weights(neural_network):
    for i in range(len(neural_network.weights_)):
        neural_network.weights_[i] = np.loadtxt("weights\\weights" + str(i + 1) + ".txt")

    print("OK")
