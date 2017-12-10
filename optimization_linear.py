import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K, Input, Model
from keras.layers import Dense

import warnings
warnings.filterwarnings('ignore')

train_x =
train_y =


#custom loss function

def custom_loss(y_ground, y_prediction):
    pass

def custom_activation(x):
    pass

input = Input((1,))
pred = Dense(input_dim=1, output_dim=1, init='uniform', activation=custom_activation)(input)

model = Model(inputs=input, outputs=pred)

model.compile(loss='mean_squared_error', optimizer='rmsprop')

weights = model.layers[1].get_weights()
print(weights)
w_init = weights[0][0][0]
b_init = weights[1][0]
print('Linear regression model is initialized with weight w: %.2f, b: %.2f' % (w_init, b_init))

# Train
model.fit(train_x, train_y, nb_epoch=1, verbose=1)

weights = model.layers[1].get_weights()
w = weights[0][0][0]
b = weights[1][0]
print('Linear regression model is trained with weight w: %.2f, b: %.2f' % (w, b))
