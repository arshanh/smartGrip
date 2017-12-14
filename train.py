""" Python script to train neural network"""
import time

start = time.time()

# Import Data
import pandas as pd

x = pd.read_csv("inputs.csv", header=None)
y = pd.read_csv("targets.csv", header=None)

# Construct scaler for standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Split data into training and testing data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33)

# Fit only to the training data
scaler.fit(x_train)
# Standardize both
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Create Neural Net (MLP with 100 neurons in hidden layer, 2000 training iterations)
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=2000, max_iter=10000)

# Train the network
mlp.fit(x_train, y_train)

# Test the network
predictions = mlp.predict(x_test)

# Report Accuracy
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, predictions))

inputs = pd.read_csv("inputs.csv", header=None)

import pickle

filename = "nn.lib"
pickle._dump(mlp, open(filename, 'wb'))

filename = "sc.lib"
pickle._dump(scaler, open(filename, 'wb'))

print('It took {0:0.1f} seconds'.format(time.time() - start))
