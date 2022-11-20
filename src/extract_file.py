import pickle
import tensorflow_hub as hub
import tensorflow as tf
from PIL import Image
import numpy as np
from glob import glob

# the shape use in this model
IMAGE_SHAPE = (224, 224)
# load the model
layer = hub.KerasLayer("feature-vector")
model = tf.keras.Sequential([layer])


# to extract features from the images
def get_features(img):
    # load image and resize to the model's standard shape
    img = Image.open(img).convert('L').resize(IMAGE_SHAPE)
    # extract the 3-dimensions
    img = np.stack((img,) * 3, axis=-1)
    # normalize the values to 0-1
    img = np.array(img) / 255.0
    # get visual representation, extract the features
    embedding = model.predict(img[np.newaxis, ...])
    feature = np.array(embedding)
    # flattened to one dimension
    flattened_feature = feature.flatten()
    return flattened_feature


# get images from database
database_dir = "image.orig/examples"
database = sorted(glob(database_dir + "/*.jpg"))
# calculate all images
for img in database:
    print(img)
    # save the features
    with open("features/"+img[19:-3]+"pkl", 'wb') as f:
        pickle.dump(get_features(img), f)
