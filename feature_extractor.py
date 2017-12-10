import numpy as np
from keras import datasets, Input, Model
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()


### Przeskaluj dane treningowe tzn. Sprowadż je do postaci wektora, o wartosciach nie przekraczajacych 1.
### Domyslnie wartosci w xtr, xtst są w zakresie 0-255, a rozmiar wynosi 28 x 28 x 1


###

nb_classes = 10

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]




def linear_classifier_model():
    """single layer, 10 output classes"""

    # create model
    input =
    x =

    model = Model(input,x)
    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


# Instantiate model
model = linear_classifier_model()

# Train model
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=1, batch_size=200, verbose=2)

# Final evaluation of the model
scores = model.evaluate(xtst, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

layer_dense_1 = model.get_layer(index=1)

print(layer_dense_1.get_weights())
weights1 = layer_dense_1.get_weights()[0]
bias1 = layer_dense_1.get_weights()[1]

# Cast shape to 2d
weights1 = weights1.reshape(28, 28, 10)

# lot the weights for the first 4 digits
for i in range(5):
    plt.subplot(1, 5, 1 + i)
    plt.imshow(weights1[:, :, i], cmap=plt.get_cmap('gray'))
plt.show()

from keras import backend as K
K.set_image_data_format('channels_last')

img_shape = (28, 28, 1)
x_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
x_test = X_test.reshape(X_test.shape[0], 28, 28, 1)