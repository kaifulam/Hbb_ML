import h5py
import numpy as np

#create input matrix
file_Sig = h5py.File("/phys/users/kaifulam/MachineLearning/dguest/h5_files/sig_jets.h5", 'r')
ds_Sig = file_Sig['clusters'][:500000]

print(ds_Sig.dtype.names)

pt = ds_Sig['pt']
eta = ds_Sig['eta']
#Gaussian Contrast Normalize pt and eta, across the full 2D array
pt = (pt - np.mean(pt)) / np.std(pt)
eta = (eta - np.mean(eta)) / np.std(eta)
merged_Sig = np.concatenate((pt[..., None], eta[..., None]), axis=2)
print("merged_Sig.shape", merged_Sig.shape)
print(merged_Sig[1,:,1])

del file_Sig, ds_Sig

file_Bg = h5py.File("/phys/users/kaifulam/MachineLearning/dguest/h5_files/bg_jets.h5", 'r')
ds_Bg = file_Bg['clusters'][:500000]

#print(ds_Bg.dtype.names)

pt = ds_Bg['pt']
eta = ds_Bg['eta']
#Gaussian Contrast Normalize pt and eta, across the full 2D array
pt = (pt - np.mean(pt)) / np.std(pt)
eta = (eta - np.mean(eta)) / np.std(eta)
merged_Bg = np.concatenate((pt[..., None], eta[..., None]), axis=2)

Training_input = np.concatenate((merged_Sig, merged_Bg), axis=0)
print("Training_input.shape", Training_input.shape)
#difference from np.vstack([merged_Sig, merged_Bg])??
#print("Training_input", Training_input[1:10, :])

np.savetxt("Training_input.csv", Training_input.flatten(), delimiter=',')
del file_Bg, ds_Bg

#setting flavor
flavor_result = np.array([True]*500000 + [False]*500000)
flavor_result = flavor_result.reshape(1000000,1)
#print("flavor_result", flavor_result[1:10])

np.savetxt("flavor_result", flavor_result, delimiter=',')

'''
#Training...
from keras.layers import GRU, Highway, Dense, Dropout, MaxoutDense, Activation, Masking
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.models import Sequential
from keras.legacy.models import Graph

model = Sequential()
#model.add(Masking(mask_value=-999, input_shape=(40, 2)))
model.add(GRU(25, input_shape=(40,2), dropout_W = 0.05))
# remove Maxout for tensorflow
model.add(MaxoutDense(64, 5))  #, input_shape=graph.nodes['dropout'].output_shape[1:]))
model.add(Dense(64, activation='relu'))

model.add(Dropout(0.4))

model.add(Highway(activation = 'relu'))

model.add(Dropout(0.3))
model.add(Dense(2))

model.add(Activation('softmax'))

print('Compiling model...')
model.compile(optimizer = 'Adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

print ('Training:')
try:
    history = model.fit(Training_input, flavor_result, batch_size=128,
        callbacks = [
            EarlyStopping(verbose=True, patience=20, monitor='val_loss'),
            ModelCheckpoint('-progress', monitor='val_loss', verbose=True, save_best_only=True)
        ],
    nb_epoch=100,
    validation_split = 0.2,
    #show_accuracy=True
    )

except KeyboardInterrupt:
    print('Training ended early.')

print("history keys", history.history.keys())

import matplotlib.pyplot as plt
# summarize history for accuracy
f1 = plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
f1.savefig("model_acc.png")

# summarize history for loss
f2 = plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
f2.savefig('model_loss.png')
'''
