from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop
from keras import backend as K, Input, Model
import warnings
warnings.filterwarnings('ignore')
batch_size = 128
nb_classes = 10
nb_epoch = 10

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape, '\n',  y_train)
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_Train = np_utils.to_categorical(y_train, nb_classes)
Y_Test = np_utils.to_categorical(y_test, nb_classes)

fig = plt.figure()
for i, img_pil in enumerate(X_train):

    img = img_pil.reshape((28, 28))
    ax = plt.subplot(1, 1, 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')

    plt.imshow(img)
    plt.show()

    if i == 6:
        plt.show()
        break

input = Input((784,))

### Przygotuj model pierwszy

x = Dense(output_dim=10,  init='normal', activation='softmax')(input)

model = Model(input,x)
model.compile(optimizer=SGD(lr=0.05), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit(X_train, Y_Train, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)

eval_md = model.evaluate(X_test, Y_Test, verbose=1)
print('Summary Loss: %.2f, Accuracy: %.2f' % (eval_md[0], eval_md[1]))






(X_train, y_train), (X_test, y_test) = mnist.load_data()
print('X_train shape:', X_train.shape)

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print('X_train shape:', X_train.shape)
print('Y_train shape:', y_train.shape, Y_train[0])
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convolutional model - przygotuj model 2 wykorzystujac filtry splotowe
input = Input(input_shape)

x = Convolution2D(32, 3, 3, border_mode='same', activation='relu') (x)
x = MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='same') (x)
x = Flatten()(x)


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


model = Model(inputs=input, outputs= x)
model.compile()

history = model.fit(X_train, Y_Train, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)

# Evaluate
evaluation = model.evaluate(X_test, Y_Test, verbose=1)
print('Summary Loss: %.2f, Accuracy: %.2f' % (evaluation[0], evaluation[1]))
