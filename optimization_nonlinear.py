import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K, Input, Model
from keras.layers import Dense

import warnings
warnings.filterwarnings('ignore')

SHAPE =
t = np.linspace(-1, 1, 256)

x_tst =
y_tst =


x_tr =
y_tr =

x_tr = np.reshape(x_tr, (x_tr.shape[0]//SHAPE,SHAPE))
x_tst = np.reshape(x_tst, (x_tst.shape[0]//SHAPE,SHAPE))

y_tr = np.reshape(y_tr, (y_tr.shape[0]//SHAPE,SHAPE))
y_tst = np.reshape(y_tst, (y_tst.shape[0]//SHAPE,SHAPE))


### Tutaj czas na eksperymenty. Dodaj nowe warstwy Dense, zmień funkcje aktywacji.

input = Input((SHAPE,))
x = Dense(output_dim = 100, activation ='relu')(input)
pred = Dense(output_dim = SHAPE, activation ='linear')(x)

model = Model(inputs=input, output=pred)
model.compile(loss='mean_squared_error', optimizer='rmsprop')


history = model.fit(x_tr, y_tr, nb_epoch=1, batch_size=32, verbose=1)

eval_md = model.evaluate(x_tst, y_tst, verbose=1)
print(eval_md)
print('Summary Loss: %.2f' % (eval_md))


out = model.predict(x_tst)
output = np.reshape(out, (out.shape[0]*out.shape[1]))
x_tst = np.reshape(x_tst, (x_tst.shape[0]*x_tst.shape[1]))
y_tst = np.reshape(y_tst, (y_tst.shape[0]*y_tst.shape[1]))

print(y_tst.shape, x_tst.shape)

### Wykres dla funkcji nielinowych ###