import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import backend as K

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd
from sklearn.model_selection import train_test_split


from pyspark.sql.functions import split, size, regexp_replace, col, when
from keras.layers import Input
from keras.layers.core import Dropout, Lambda

from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd

# Keras / Deep Learning
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras import optimizers, regularizers
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

import numpy as np
import os
from glob import glob
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import pyspark
from pyspark.sql import SparkSession


def build(width, height, depth):
    inputs = Input((height, width, depth))
    s = Lambda(lambda x: x / 255)(inputs)

    c1 = keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)

    outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c2)

    model = Model(inputs=[inputs], outputs=[outputs])

    # return the constructed network architecture
    return model


def create_compile_model():
    model = keras.models.Sequential([
        # Convolution2D(32, kernel_size=(3, 3), activation='relu', input_shape=(286, 384, 1)),
        # keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu',
        #                     input_shape=(227, 227, 3)),
        keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                            input_shape=(227, 227, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(12, activation='softmax')
    ])

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'Recall']
    )
    return model


def extract_label(path):
    label = split(path, '/')
    length = size(label)
    return label[length - 2]


if __name__ == '__main__':
    setattr(tf.compat.v1.nn.rnn_cell.GRUCell, '_deepcopy_', lambda self, _: self)
    setattr(tf.compat.v1.nn.rnn_cell.BasicLSTMCell, '_deepcopy_', lambda self, _: self)
    setattr(tf.compat.v1.nn.rnn_cell.MultiRNNCell, '_deepcopy_', lambda self, _: self)

    spark = SparkSession.builder.master("local[1]") \
        .appName('fruitrecognition') \
        .getOrCreate()

    # path to your image source directory
    img_dir = '/Users/jane/Documents/workspace/FruitRecognition/data/test/**'
    # Read image data using new image scheme
    image_df = spark.read.format("image").load(img_dir)
    image_df

    selected_df = image_df.select("image.origin")
    print(type(selected_df))

    new_df = selected_df.withColumn('label', selected_df.origin)

    # df.withColumn('year', split(df['dob'], '-').getItem(0))
    new_df = new_df.withColumn('label', extract_label(new_df['label']))

    new_df.show()
    # selected_df.withColumn("label", selected_df.origin.split('/')[-2])

    pd_df = new_df.toPandas()

    a = pd_df['origin'][0]
    print(a)
    pd_df['path'] = pd_df.apply(lambda row: row.origin.split('//')[1], axis=1)
    pd_df['label'].unique()

    train_set, test_set = train_test_split(pd_df, test_size=0.2, random_state=17)
    train_set.shape
    test_set.shape
    train_set.head()
    print(train_set['origin'][0])
    train_set

    # Generating Spark Context
    X_train = train_set['origin']
    y_train = train_set['label']

    sc = spark.sparkContext
    rdd = to_simple_rdd(sc, X_train, y_train)

    # model = create_compile_model()
    model = build(227, 227, 3)
    opt = tf.optimizers.Adam()
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    setattr(tf.compat.v1.nn.rnn_cell.GRUCell, '_deepcopy_', lambda self, _: self)
    setattr(tf.compat.v1.nn.rnn_cell.BasicLSTMCell, '_deepcopy_', lambda self, _: self)
    setattr(tf.compat.v1.nn.rnn_cell.MultiRNNCell, '_deepcopy_', lambda self, _: self)

    # adagrad = elephas_optimizers.Adagrad()
    spark_model = SparkModel(model, frequency='epoch', mode='asynchronous', num_workers=2, batch_size=32)
    spark_model.fit(rdd, epochs=40, batch_size=32, verbose=0, validation_split=0.1)
    print("Done!!!")
