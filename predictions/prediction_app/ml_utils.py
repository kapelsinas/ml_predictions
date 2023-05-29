# ml_utils.py
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Get the current directory of the ml_utils.py file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the model file
model_path = os.path.join(current_dir, 'skinModel', 'model.hdf5')


irv2 = tf.keras.applications.InceptionResNetV2(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classifier_activation="softmax",
)

# Exclude the last 28 layers of the model.
conv = irv2.layers[-28].output

conv  = Activation('relu')(conv)
conv = Dropout(0.5)(conv)

output = Flatten()(conv)
output = Dense(7, activation='softmax')(output)
model = Model(inputs=irv2.input, outputs=output)

opt1=tf.keras.optimizers.Adam(learning_rate=0.01,epsilon=0.1)
model.compile(optimizer=opt1,
             loss='categorical_crossentropy',
             metrics=['accuracy'])

class_weights = {   
                    0: 1.0,  # akiec
                    1: 1.0,  # bcc
                    2: 1.0,  # bkl
                    3: 1.0,  # df
                    4: 4.0,  # mel
                    5: 1.0,  # nv
                    6: 1.0,  # vasc
                }

checkpoint=  ModelCheckpoint(filepath = 'saved_model.hdf5',monitor='val_accuracy',save_best_only=True,save_weights_only=True)

Earlystop = EarlyStopping(monitor='val_loss', mode='min',patience=30, min_delta=0.001)

# Load the saved weights
model.load_weights(model_path)

def preprocess(data):
    # Resize the image to 299x299
    resized_image = np.resize(data, (299, 299, 3))

    # Convert the image to a numpy array
    image_array = img_to_array(resized_image)

    # Expand the dimensions of the image array to fit the model input shape
    expanded_image_array = np.expand_dims(image_array, axis=0)

    # Apply any preprocessing specific to the model
    # Use the InceptionResNetV2 preprocess_input function
    preprocessed_image_array = preprocess_input(expanded_image_array)

    return preprocessed_image_array


def process_results(prediction):
    class_mapping = {
        0: 'Actinic keratoses',
        1: 'Basal cell carcinoma',
        2: 'Benign keratosis-like lesions',
        3: 'Dermatofibroma',
        4: 'Melanocytic nevi',
        5: 'Melanoma',
        6: 'Vascular lesions',
    }

    # Get probabilities as a list
    probabilities = prediction[0].tolist()

    # Create a dictionary with class names and probabilities
    class_probabilities = {
        class_mapping[i]: prob for i, prob in enumerate(probabilities)}

    # Sort the dictionary by percentage in descending order
    sorted_class_probabilities = {k: v for k, v in sorted(
        class_probabilities.items(), key=lambda item: item[1], reverse=True)}

    return sorted_class_probabilities


def predict(data):
    # Preprocess your input data here, if necessary
    preprocessed_data = preprocess(data)

    # Load your ML model here

    # Make a prediction using the ML model
    prediction = model.predict(preprocessed_data)

    # Process the prediction results and return them
    return process_results(prediction)
