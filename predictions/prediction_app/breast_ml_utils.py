# ml_utils.py
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

# Get the current directory of the ml_utils.py file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the model file
model_path = os.path.join(current_dir, 'breastModel', 'model.hdf5')

def create_model():

    pre_trained_model = InceptionResNetV2(input_shape=(299, 299, 1),
                                          include_top=False,
                                          weights=None)
    for layer in pre_trained_model.layers:
        layer.trainable = False

    last_output = pre_trained_model.output

    # Flatten the output layer to 1 dimension
    x = Flatten()(last_output)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = Dense(1024, activation='relu')(x)
    # Add a dropout rate of 0.2
    x = Dropout(0.2)(x)
    # Add a final sigmoid layer for classification
    x = Dense(1, activation='sigmoid')(x)

    model = Model(pre_trained_model.input, x)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=10000,
        decay_rate=0.9)

    optimizer = Adam(learning_rate=lr_schedule)
    # Loss
    loss = tf.keras.losses.BinaryCrossentropy()

    # Metrics
    metrics = [tf.keras.metrics.Precision(name='prec'),
               tf.keras.metrics.Recall(name='rec'),
               tf.keras.metrics.BinaryAccuracy(name='BinAcc'),
               ]

    # Model compilation
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # Load the saved weights
    model.load_weights(model_path, by_name=True)

    return model
