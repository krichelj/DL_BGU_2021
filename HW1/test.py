from neural_net import *
import tensorflow as tf
import time

# keras mnist loading code
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data('./data')
x_train = x_train.reshape(x_train.shape[0], -1).T / 255
x_test = x_test.reshape(x_test.shape[0], -1).T / 255

print('-'*40 + f'batch size = {64}' + '-'*40)
training_start_time = time.time()
parameters, costs = L_layer_model(x_train, y_train,
                                  layers_dims=[784, 20, 7, 5, 10],
                                  learning_rate=0.009,
                                  num_iterations=10,
                                  batch_size=64)

time_hh_mm_ss = time.strftime('%M:%S', time.gmtime(time.time() - training_start_time))
print(f'Total training time: {time_hh_mm_ss}')

test_acc = Predict(x_test, y_test, parameters)
print(f'Last training step test accuracy: {test_acc*100:.5f}%')