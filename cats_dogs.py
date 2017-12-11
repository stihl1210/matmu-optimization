import PIL

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import load_model

import numpy as np

import scipy

def deprocess_image(path):
    x = np.asarray(PIL.Image.open(path))
    x = scipy.misc.imresize(x, (350,350))
    # x = np.divide(x, 255.0)
    # x = np.subtract(x, 1.0)
    # x = np.multiply(x, 2.0)
    x = np.expand_dims(x, axis=0)
    print(x.shape)
    return x

model = load_model('best_model.h5')

img = deprocess_image('./train/dog.66.jpg')

pred = model.predict(img)

print(pred, 1)

# cat - 0, dog - 1