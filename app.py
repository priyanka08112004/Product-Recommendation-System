import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tqdm import tqdm
import numpy as np
from numpy.linalg import norm
import os
import pickle

# Load ResNet50 model with pretrained weights
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Create a sequential model with ResNet50 and a global max pooling layer
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def extract_features(img_path, model):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

# Collect image file names
filenames = []
for file in os.listdir('images'):
    filenames.append(os.path.join('images', file))

# Extract features and save them
feature_list = []
for file in tqdm(filenames):
    feature_list.append(extract_features(file, model))

# Save the feature list and filenames using pickle
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))
