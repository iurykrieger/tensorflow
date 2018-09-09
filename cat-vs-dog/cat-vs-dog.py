import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from random import shuffle
from tqdm import tqdm
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

TRAIN_DIR = 'train'
TEST_DIR = 'test'
IMG_WIDTH = 50
IMG_HEIGHT = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.3
CATEGORIES = ['dog', 'cat']
DENSE_LAYERS = [0]
LAYER_SIZES = [64]
CONV_LAYERS = [3]

def normalize_image(path):
    try: 
        return cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_WIDTH, IMG_HEIGHT))
    except Exception as e:
        return None

def normalize_train_data():
    train_data = []
    for label in os.listdir(TRAIN_DIR):
        path = os.path.join(TRAIN_DIR, label)
        print('\nNormalizing {name} data...'.format(name=label))
        for img in tqdm(os.listdir(path)):
            img_path = os.path.join(path, img)
            normalized_image = normalize_image(img_path)
            if normalized_image is not None:
                train_data.append([normalized_image, CATEGORIES.index(label)])

    shuffle(train_data)
    train_features = []
    train_labels = []
    for features, label in train_data:
        train_features.append(features)
        train_labels.append(label)
    train_features = np.array(train_features).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1) / 255.0
    train_labels = np.array(train_labels)
    return (train_features, train_labels)

def get_train_data():
    try:
        train_X = np.load('train_X.npy')
        train_y = np.load('train_y.npy')
        return (train_X, train_y)
    except Exception as e:
        train_X, train_y = normalize_train_data()
        np.save('train_X.npy', train_X)
        np.save('train_y.npy', train_y)
        return (train_X, train_y)

def get_cnn_model(data_shape, dense_layer, conv_layer, layer_size):
    model = Sequential()
    # 1st layer
    model.add(Conv2D(layer_size, (3,3), input_shape=data_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for layer in range(conv_layer - 1):
        model.add(Conv2D(layer_size, (3,3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    for layer in range(dense_layer):
        model.add(Dense(layer_size))
        model.add(Activation('relu'))
    
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_cnn_model(model_name, train_X, train_y,  dense_layer, conv_layer, layer_size, epochs=3):
    tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))
    model = get_cnn_model(train_X.shape[1:],  dense_layer, conv_layer, layer_size)
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        batch_size=32,
        validation_split=VALIDATION_SPLIT,
        callbacks=[tensorboard]
    )
    return model

def train_data():
    train_X, train_y = get_train_data()

    for dense_layer in DENSE_LAYERS:
        for layer_size in LAYER_SIZES:
            for conv_layer in CONV_LAYERS:
                model_name = '{}-conv-{}-dense-{}-nodes.model'.format(conv_layer, dense_layer, layer_size)
                model = train_cnn_model(model_name, train_X, train_y, dense_layer, conv_layer, layer_size, 12)
                model.save(model_name)

model = load_model('3-conv-0-dense-64-nodes.model')
image = normalize_image('test/cat7.jpg')
image = np.array([image]).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1) / 255.0
prediction = model.predict(image)
print(prediction[0][0])
print(CATEGORIES[int(round(prediction[0][0]))])