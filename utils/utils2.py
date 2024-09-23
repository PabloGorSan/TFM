import sqlite3
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib
import os
import sys
from collections import Counter
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input, BatchNormalization, Activation, Add, LeakyReLU
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import layers
from tensorflow.keras.applications import InceptionResNetV2, VGG16, MobileNetV2
from tensorflow.keras.utils import to_categorical
from PIL import Image
import requests
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output as cls
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import pandas as pd
import PIL

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import statistics
import ast
import json
import statistics

import glob
import cv2
from PIL import Image
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_path, read_size, batch_size, num_classes, csv_length, initial_pos = True, final_pos = False, legal_moves = False, move = False, single_move = False, single_and_partial = False, shuffle = True, sparse_single_and_partial = False, pgn_data = False):
        self.file_path = file_path
        self.batch_size = batch_size
        self.read_size = read_size
        self.num_classes = num_classes
        self.csv_length = csv_length
        self.current_read = None
        self.initial_pos = initial_pos
        self.final_pos = final_pos
        self.legal_moves = legal_moves
        self.move = move
        self.single_move = single_move
        self.single_and_partial = single_and_partial
        self.sparse_single_and_partial = sparse_single_and_partial
        self.pgn_data = pgn_data
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.csv_length / self.batch_size))

    def on_epoch_end(self):
        pass

    def __getitem__(self, index):
        if index % (self.read_size // self.batch_size) == 0:
            self.current_read = pd.read_csv(self.file_path, skiprows=range(1, (index * self.batch_size) + 1), nrows=self.read_size)
            if self.shuffle:
                self.current_read = self.current_read.sample(frac=1).reset_index(drop=True)
        batch = self.get_batch(index % (self.read_size // self.batch_size))
        X, y = self.__data_generation__(batch)
        return X, y
    

    def __data_generation__(self, df):
        if self.pgn_data:
            X = np.array(df['board_pgn'])
        else:
            X = np.array(df['board'].apply(json.loads).tolist())
        if self.initial_pos and not self.final_pos:
            y = self.encode_pos(df['initial_pos'])
        elif self.final_pos and not self.initial_pos:
            y = self.encode_pos(df['final_pos'])
        elif self.initial_pos and self.final_pos:
            y = [self.encode_pos(df['initial_pos']), self.encode_pos(df['final_pos'])]
        if self.move:
            y = self.encode_move_one_hot(df['move'])
        if self.single_move:
            y = self.encode_single_move(df['move'])
        if self.single_and_partial:
            y = [self.encode_single_move(df['move']), self.encode_pos(df['initial_pos']), self.encode_pos(df['final_pos'])]
        if self.sparse_single_and_partial:
            y = [self.encode_sparse_single_move(df['move']), self.encode_pos(df['initial_pos']), self.encode_pos(df['final_pos'])]
        if self.pgn_data:
            y = [self.encode_sparse_single_move(df['move']), self.encode_pos(df['initial_pos']), self.encode_pos(df['final_pos'])]
        return X, y

    
    def get_batch(self, index):
        start = index * self.batch_size
        end = start + self.batch_size
        return self.current_read.iloc[start:end]

    def encode_pos(self, positions):
        column_to_number = {
            'a': 1,
            'b': 2,
            'c': 3,
            'd': 4,
            'e': 5,
            'f': 6,
            'g': 7,
            'h': 8
        }
        encoded_positions = []
        for pos in positions:
            encoded_positions.append((column_to_number.get(pos[0]) - 1) +  ((8 - int(pos[1])) * 8))
        return np.array(encoded_positions)
    
    def encode_move_one_hot(self, moves):
        column_to_number = {
            'a': 1,
            'b': 2,
            'c': 3,
            'd': 4,
            'e': 5,
            'f': 6,
            'g': 7,
            'h': 8
        }
        encoded_moves = []
        for move in moves:
            encoded_initial_pos = np.zeros(64)
            encoded_initial_pos[((column_to_number.get(move[0]) - 1)) +  ((8 - int(move[1])) * 8)] = 1
            encoded_final_pos = np.zeros(64)
            encoded_final_pos[((column_to_number.get(move[2]) - 1)) +  ((8 - int(move[3])) * 8)] = 1

            encoded_moves.append(np.concatenate((encoded_initial_pos, encoded_final_pos), axis=None))
        return np.array(encoded_moves)
    
    def encode_single_move(self, moves):
        column_to_number = {
            'a': 1,
            'b': 2,
            'c': 3,
            'd': 4,
            'e': 5,
            'f': 6,
            'g': 7,
            'h': 8
        }
        encoded_moves = []
        for move in moves:
            encoded_move = np.zeros(4096)
            encoded_move[(((column_to_number.get(move[0]) - 1)) + ((8 - int(move[1])) * 8)) * 64 + (((column_to_number.get(move[2]) - 1)) + ((8 - int(move[3])) * 8))] = 1
            encoded_moves.append(encoded_move)
        return np.array(encoded_moves)
    
    def encode_sparse_single_move(self, moves):
        column_to_number = {
            'a': 1,
            'b': 2,
            'c': 3,
            'd': 4,
            'e': 5,
            'f': 6,
            'g': 7,
            'h': 8
        }
        encoded_moves = []
        for move in moves:
            encoded_moves.append((((column_to_number.get(move[0]) - 1)) + ((8 - int(move[1])) * 8)) * 64 + (((column_to_number.get(move[2]) - 1)) + ((8 - int(move[3])) * 8)))
        return np.array(encoded_moves)


    

def plot_graph(history):
  df = pd.DataFrame(history.history)

  plt.style.use('fast')
  plt.figure(figsize=(25,8))

  plt.subplot(1,2,1)
  plt.title("Loss Curve")
  plt.plot(df['loss'], label="Loss")
  plt.plot(df['val_loss'], label="Val Loss")
  plt.xlabel("Epochs")
  plt.legend(fontsize=15)
  plt.ylabel("Loss")

  plt.subplot(1,2,2)
  plt.title("Accuracy Curve")
  plt.plot(df['accuracy'], label="Accuracy")
  plt.plot(df['val_accuracy'], label="Val Accuracy")
  plt.xlabel("Epochs")
  plt.legend(fontsize=15)
  plt.ylabel("Accuracy")

  plt.show()
  
def bar_plot(x, y):
    plt.figure(figsize=(15, 6))
    plt.bar(x, y)
    plt.title('Moves predicted')
    plt.xlabel('Move')
    plt.ylabel('Count')
    plt.xticks(x, [f'{i}' for i in x])
    plt.tight_layout()
    plt.show()

def two_bar_plot(x, y1, y2):
    bar_width = 0.35
    indices = np.arange(len(x))
    plt.figure(figsize=(15, 6))
    plt.bar(indices, y1, bar_width, label='Predicted')
    plt.bar(indices + bar_width, y2, bar_width, label='Ground Truth')
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Bar Plot with Two Sets of Bars')
    plt.xticks(indices + bar_width / 2, x)
    plt.legend()
    plt.tight_layout()
    plt.show()


class StepwiseLogger(keras.callbacks.Callback):
    def __init__(self, val_generator, steps_per_epoch_val, interval, divide=10):
        super().__init__()
        self.val_generator = val_generator
        self.steps_per_epoch_val = steps_per_epoch_val
        self.interval = interval
        self.divide = divide
        self.history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

    def on_train_batch_end(self, batch, logs=None):
        if self.model.stop_training:
            return

        if (batch + 1) % self.interval == 0:
            val_logs = self.model.evaluate(self.val_generator, steps=self.steps_per_epoch_val/self.divide, verbose=0)
            self.history['loss'].append(logs.get('loss'))
            self.history['accuracy'].append(logs.get('accuracy'))
            self.history['val_loss'].append(val_logs[0])
            self.history['val_accuracy'].append(val_logs[1])

def decode_pos(positions):
        number_to_column = {
            1 : 'a',
            2 : 'b',
            3 : 'c',
            4 : 'd',
            5 : 'e',
            6 : 'f',
            7 : 'g',
            8 : 'h'
        }
        decoded_positions = []
        for pos in positions:
            decoded_pos = number_to_column.get(pos % 8 + 1) + str(8 - (pos // 8))
            decoded_positions.append(decoded_pos)
        return np.array(decoded_positions)


class ErrorDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_path, read_size, batch_size, csv_length):
        self.file_path = file_path
        self.batch_size = batch_size
        self.read_size = read_size
        self.csv_length = csv_length
        self.current_read = None
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.csv_length / self.batch_size))

    def on_epoch_end(self):
        pass

    def __getitem__(self, index):
        if index % (self.read_size // self.batch_size) == 0:
            self.current_read = pd.read_csv(self.file_path, skiprows=range(1, (index * self.batch_size) + 1), nrows=self.read_size)
        batch = self.get_batch(index % (self.read_size // self.batch_size))
        X, y = self.__data_generation__(batch)
        return X, y

    def __data_generation__(self, df):
        X = np.array(df['board'].apply(json.loads).tolist())
        y = np.array(df['is_blunder_cp'])
        return X, y
    
    def get_batch(self, index):
        start = index * self.batch_size
        end = start + self.batch_size
        return self.current_read.iloc[start:end]
