from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import load_model
import matplotlib.pyplot as plt

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

img = load_img('PetImages/Cat/2.jpg')
x = img_to_array(img)
print(x.shape)
x = x.reshape((1,) + x.shape)

import os
directory_test = './test-generator2'

if not os.path.exists(directory_test):
    os.makedirs(directory_test)

i = 0

for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=directory_test,
                          save_prefix='cat',
                          save_format='jpg'):
    i += 1
    if i > 20:
        break

names = [na for na in os.listdir(directory_test)]
fig = plt.figure()
for img_pil in names:
    img = load_img(os.path.join(directory_test, img_pil))
    x = img_to_array(img)

    ax = plt.subplot(1, 1, 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')

    plt.imshow(img)
    plt.show()

    if i == 6:
        plt.show()
        break

model = load_model('catsdogs.h5')

x = model.predict(img)

