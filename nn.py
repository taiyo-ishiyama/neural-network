import numpy as np
import time
import matplotlib.pyplot as plt
import sys

def read_input_file(file_path): # read input file and process data
  data = np.loadtxt(file_path, delimiter=',', skiprows=1)
  x = data[:, 1:] / 255.0  # convert the pixel values
  y = data[:, 0].astype(int) # get data lable
  y_onehot_encode = np.eye(10)[y]  # convert data labels to one-hot encoding

  return x, y_onehot_encode

def sigmoid(x): # activate function for forward propagation
  return 1/(1 + np.exp(-x))

def calculate_delta_output(y, d): # calculate error for output layer
  return y * (1 - y) * (d - y)

def calculate_delta_hidden(y, w, previous_delta): # calculate error for hidden layer
  return y * (1 - y) * np.dot(w.T, previous_delta)

def initialize_network(input_layer, hidden_layer, output_layer): # intialise weight and bias
  network = {}
  # obtain from standard normal
  network["w1"] = np.random.randn(hidden_layer, input_layer)
  network["w2"] = np.random.randn(output_layer, hidden_layer)
  network["b1"] = np.random.randn(hidden_layer)
  network["b2"] = np.random.randn(output_layer)

  return network

def forward_propagation(network, x): # handle forward propagation
  W1, W2= network["w1"], network["w2"]
  B1, B2 = network["b1"], network["b2"]

  u1 = np.dot(W1,x) + B1 # process data for input layer
  z1 = sigmoid(u1) # use activate function to get output

  u2 = np.dot(W2,z1) + B2 # process data for hidden layer
  z2 = sigmoid(u2) # use activate function to get output
  y = z2

  return y, z1


def back_propagation(network, x, d, z1, y, r): # handle back propagation
  W1, W2 = network["w1"], network["w2"]
  B1, B2 = network["b1"], network["b2"]

  delta2 = calculate_delta_output(y, d) # compute errors
  network["b2"] += r * np.sum(delta2, axis=0) # update bias between output and hidden layer
  network["w2"] += r * np.outer(delta2, z1) # update weight between output and hidden layer

  delta1 = calculate_delta_hidden(z1, W2, delta2) # compute errors
  network["b1"] += r * np.sum(delta1, axis=0) # update bias between hidden and input layer
  network["w1"] += r * np.outer(delta1, x) # update weight between hidden and input layer

  return network

# train network for given learning rate, batch size, and epochs
def train_network(x_train_data, y_train_data, x_test_data, y_test_data, r, batch_size, epochs, input_layer, hidden_layer, output_layer):
  network = initialize_network(input_layer, hidden_layer, output_layer)

  train_loss_list = [] # store loss for each epoch
  train_accuracy_list = [] # store train data accuracy for each epoch
  test_accuracy_list = [] # store test data accuracy for each epoch

  for current_epoch in range(epochs): # repeat the training for the number of epoch
    loss_in_epoch = 0
    correct_data = 0
    for i in range(0, len(x_train_data), batch_size): # implement training for each batch
      x_current_batch = x_train_data[i:i+batch_size]
      y_current_batch = y_train_data[i:i+batch_size]

      for x, d in zip(x_current_batch, y_current_batch):
        y, z1 = forward_propagation(network, x)
        network = back_propagation(network, x, d, z1, y, r)
        loss_in_epoch += np.sum((d - y) ** 2) / 2  # compute mean squared error
        if np.argmax(y) == np.argmax(d): # get number of correct data
            correct_data += 1

    train_loss_list.append(loss_in_epoch / len(x_train_data))
    train_accuracy_list.append(correct_data / len(x_train_data) * 100)

    correct_test_data = 0
    total_test_data = 0
    for x, d in zip(x_test_data, y_test_data): # use the testing dataset
      y, _ = forward_propagation(network, x)
      if np.argmax(y) == np.argmax(d):
        correct_test_data += 1
      total_test_data += 1
    test_accuracy = correct_test_data / total_test_data * 100
    test_accuracy_list.append(test_accuracy)

    print(f"Epoch {current_epoch+1}/{epochs}, Loss: {loss_in_epoch / len(x_train_data):.2f}, Accuracy for training data: {correct_data / len(x_train_data) * 100:.2f}%, Accuracy for testing data: {test_accuracy:.2f}%")

  return train_accuracy_list, test_accuracy_list


input_layer = int(sys.argv[1])
hidden_layer = int(sys.argv[2])
output_layer = int(sys.argv[3])
train_file_name = sys.argv[4]
test_file_name = sys.argv[5]

x_train_data, y_train_data = read_input_file(train_file_name)
x_test_data, y_test_data = read_input_file(test_file_name)


# Test 1
r = 3
batch_size = 20
epochs = 30
train_accuracy_list, test_accuracy_list = train_network(x_train_data, y_train_data, x_test_data, y_test_data, r, batch_size, epochs, input_layer, hidden_layer, output_layer)
print(f"Maximum accuracy (learning rate: {r}, batch size: {batch_size} epoch: {epochs}: {max(test_accuracy_list):.2f}%")

plt.figure(figsize=(12, 5))
plt.plot(test_accuracy_list, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()


# # Test 2
r_list = [0.001, 0.01, 1.0, 10, 100]
batch_size = 20
epochs = 30
color_list = ['b', 'g', 'r', 'c', 'm']

plt.figure(figsize=(12, 5))

for r, color in zip(r_list, color_list):
  train_accuracy_list, test_accuracy_list = train_network(x_train_data, y_train_data, x_test_data, y_test_data, r, batch_size, epochs, input_layer, hidden_layer, output_layer)
  print(f"Maximum accuracy (learning rate: {r}, batch size: {batch_size} epoch: {epochs}: {max(test_accuracy_list):.2f}%\n")
  plt.plot(test_accuracy_list, label=f'LR={r}', color=color)

plt.xlabel('Epoch')
plt.ylabel('Test Accuracy (%)')
plt.legend()
plt.show()


# Test 3
batch_size_list = [1, 5, 20, 100, 300]
r = 3
epochs = 30
time_list = []
max_accuracy_list = []

for batch_size in batch_size_list:
  start = time.time()
  _, test_accuracy_list = train_network(x_train_data, y_train_data, x_test_data, y_test_data, r, batch_size, epochs, input_layer, hidden_layer, output_layer)
  end = time.time()
  print(f"Maximum accuracy (learning rate: {r}, batch size: {batch_size} epoch: {epochs}: {max(test_accuracy_list):.2f}%")
  search_time = end - start
  print("Prcoess time:", search_time, "seconds\n")
  time_list.append(search_time)
  max_accuracy_list.append(max(test_accuracy_list))

print(f"batch size = {batch_size_list[max_accuracy_list.index(max(max_accuracy_list))]} achieved the maximum accuracy ({max(max_accuracy_list):.2f}%)")
print(f"batch size = {batch_size_list[time_list.index(max(time_list))]} took the longest process time ({max(time_list):.2f} seconds)")

plt.figure(figsize=(12, 5))
plt.plot(batch_size_list, max_accuracy_list, marker='o')
plt.xlabel('Mini-Batch Size')
plt.ylabel('Maximum Test Accuracy (%)')
plt.show()


# Test 4
r_list = [0.01, 0.01, 0.01, 0.01]
batch_size_list = [1, 5, 1, 5]
epochs_list = [30, 30, 50, 50]
max_accuracy_list = []

for r_value, batch_size_value, epochs_value in zip(r_list, batch_size_list, epochs_list):
  train_accuracy_list, test_accuracy_list = train_network(x_train_data, y_train_data, x_test_data, y_test_data, r_value, batch_size_value, epochs_value, input_layer, hidden_layer, output_layer)
  print(f"Maximum accuracy (learning rate: {r_value}, batch size: {batch_size_value} epoch: {epochs_value}: {max(test_accuracy_list):.2f}%\n")
  max_accuracy_list.append(max(test_accuracy_list))

max_index = max_accuracy_list.index(max(max_accuracy_list))
print(f"[learning rate = {r_list[max_index]}, batch size = {batch_size_list[max_index]}, epochs = {epochs_list[max_index]} achieved the maximum accuracy ({max(max_accuracy_list):.2f}%)")