from glob import glob
import cv2
import pickle
import numpy as np

database_dir = "image.orig"
database = sorted(glob(database_dir + "/*.jpg"))
detector = cv2.SIFT_create()
dic = {}

for img in database:
    img_gray = cv2.imread(img)
    _, descriptor = detector.detectAndCompute(img_gray, None)
    descriptor /= (descriptor.sum(axis=1, keepdims=True) + 1e-7)
    descriptor = np.sqrt(descriptor)
    print(img[11:])
    with open("descriptors/"+img[11:-3]+"pkl", 'wb') as f:
        pickle.dump(descriptor, f)


