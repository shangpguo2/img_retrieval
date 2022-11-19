import pickle
import tensorflow_hub as hub
import tensorflow as tf
from PIL import Image
import numpy as np
from glob import glob

IMAGE_SHAPE = (224, 224)

layer = hub.KerasLayer("feature-vector")
model = tf.keras.Sequential([layer])


def extract(file):
    # load image and resize to the model's standard shape
    file = Image.open(file).convert('L').resize(IMAGE_SHAPE)
    # extract the 3-dimensions
    file = np.stack((file,) * 3, axis=-1)
    # normalize the values to 0-1
    file = np.array(file) / 255.0
    # get visual representation, extract the features
    embedding = model.predict(file[np.newaxis, ...])
    feature = np.array(embedding)
    # flattened to one dimension
    flattened_feature = feature.flatten()
    return flattened_feature


database_dir = "image.orig/examples"
database = sorted(glob(database_dir + "/*.jpg"))
for img in database:
    print(img)
    with open("features/"+img[19:-3]+"pkl", 'wb') as f:
        pickle.dump(extract(img), f)