"""
Created on 2018-06-04
# =============================================================================
# Model compilation algorythm in python
# =============================================================================
@author: %(Drakael)s
"""
from keras.models import load_model
from keras.models import Sequential
from keras.models import Model

autoencoder = load_model('Devanagari_autoencoder.h5')

encoder = Model(inputs=autoencoder.input,
                outputs=autoencoder.get_layer('encoder').output)

dense = load_model('Devanagari_dense.h5')

model = Sequential()
model.add(encoder)
model.add(dense)
model.save('Devanagari_CAE.h5')
