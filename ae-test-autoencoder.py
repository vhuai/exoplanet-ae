#!/usr/bin/python3

# attempts to create an autoencoder

# used anaconda with a tensorflow env
# additional packages downloaded:
# tensorflow, sklearn

# from astroquery.mast import Tesscut
# from astropy.table import Table
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.datasets import fashion_mnist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle

# loading the data from pickle file created from ae-simulate-trainingdata.py
training_data = pickle.load(open("train.p", "rb"))
training_data = tf.cast(training_data, tf.float32)

test_data = pickle.load(open("test.p", "rb"))
test_data = tf.cast(test_data, tf.float32)

test_data_label = pickle.load(open("test_label.p", "rb"))
# test_data_label = np.asarray(pickle.load(open("test_label.p", "rb")), dtype=object).astype(np.float32)

class AnomalyDetector(Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(16, activation="relu")])

        self.decoder = tf.keras.Sequential([
        layers.Dense(16, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(512, activation="relu"),
        # layers.Dense(2048, activation="relu"),
        layers.Dense(36414, activation="sigmoid")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = AnomalyDetector()
autoencoder.compile(optimizer='adam', loss='mae')

history = autoencoder.fit(training_data, training_data, 
          epochs=10, 
          batch_size=50,
          validation_data=(training_data, training_data),
          shuffle=True)

reconstructions = autoencoder.predict(training_data)
train_loss = tf.keras.losses.mae(reconstructions, training_data)

plt.hist(train_loss[None,:], bins=50)
plt.xlabel("Train loss")
plt.ylabel("Number of examples")
plt.show()

threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)

def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  return tf.math.less(loss, threshold)

def print_stats(predictions, labels):
  print("Accuracy = {}".format(accuracy_score(labels, predictions)))
  print("Precision = {}".format(precision_score(labels, predictions)))
  print("Recall = {}".format(recall_score(labels, predictions)))

# broke here, bet it was due to the difference in size between training and test light curves
preds = predict(autoencoder, test_data, threshold)
print_stats(preds, test_data_label)