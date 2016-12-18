import numpy as np

#First 5000 rows: signal; last 5000 rows: background

Training_input = np.genfromtxt("Training_input2_10000.csv", delimiter=',')
Training_input = np.reshape(Training_input, (10000,40,5))
print("Training_input.shape", Training_input.shape)

#setting flavor_result [10000, 2]. The beginning 5000 rows are signals, the remaining 5000 are bckground
flavor_result = np.zeros((10000,2))
flavor_result[0:4999] = np.asarray([0,1])
flavor_result[5000:9999] = np.asarray([1,0])
#flavor_result = np.genfromtxt("flavor_result2_10000.csv", delimiter=',')
print("flavor_result.shape", flavor_result.shape)

#split the training set into training and test sets
from random import shuffle
n_rows = Training_input.shape[0]
train_rows, test_rows = int(n_rows*8/10), int(n_rows*2/10)
indx = range(n_rows)
shuffle(indx)
X_train, X_test = Training_input[indx[:train_rows], ...], Training_input[indx[train_rows : train_rows+test_rows], ...]
y_train, y_test = flavor_result[indx[:train_rows], ...], flavor_result[indx[train_rows : train_rows+test_rows], ...]
print("dimensions", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Training...
###
from keras.layers import GRU, Highway, Dense, Dropout, MaxoutDense, Activation, Masking
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.models import Sequential
from keras.legacy.models import Graph

model = Sequential()
#model.add(Masking(mask_value=-999, input_shape=(40, 2)))
model.add(GRU(25, input_shape=(40,5), dropout_W = 0.05))
# remove Maxout for tensorflow
model.add(MaxoutDense(64, 5))  #, input_shape=graph.nodes['dropout'].output_shape[1:]))
model.add(Dense(64, activation='relu'))

model.add(Dropout(0.4))

model.add(Highway(activation = 'relu'))

model.add(Dropout(0.3))
model.add(Dense(2))

model.add(Activation('softmax'))

print('Compiling model...')
model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
model.summary()

print ('Training:')
try:
    history = model.fit(X_train, y_train, batch_size=128,
        callbacks = [
            EarlyStopping(verbose=True, patience=20, monitor='val_loss'),
            ModelCheckpoint('-progress', monitor='val_loss', verbose=True, save_best_only=True)
        ],
    nb_epoch=20,
    validation_split = 0.2,
    #show_accuracy=True
    )

except KeyboardInterrupt:
    print('Training ended early.')

print("history keys", history.history.keys())

predictions = model.predict(X_test, batch_size=128)
print("predictions.shape", predictions.shape, predictions)

#np.savetxt("y_test.txt", y_test)
#np.savetxt("predictions.txt", predictions)

from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(y_test.ravel(), predictions.ravel())
roc_auc = metrics.auc(fpr, tpr)
print("fpr", fpr.shape, fpr)
print("tpr", tpr.shape, tpr)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, color='g', linewidth = 10)
plt.plot([0,1], [0,1], linestyle='--', color='k')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.show()
plt.savefig("ROC_demo", format='png')
plt.close()

'''# summarize history for accuracy
f1 = plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
#f1.savefig("model_acc.png")
plt.show()

# summarize history for loss
f2 = plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#f2.savefig('model_loss.png')
plt.show()'''
