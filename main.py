#%% load mnist dataset
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
#from keras_visualizer import visualizer

(trainX, trainY), (testX, testY) = mnist.load_data()

#%% preview data
for i in range(9):
    pyplot.subplot(331 + i)
    pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()
print(f'Train: X={trainX.shape}, Y={trainY.shape}')
print(f'Test: x={testX.shape}, y={testY.shape}')

#%% prepare data
# reshape dataset to have a single channel
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))
# one hot encode target values
trainY = to_categorical(trainY)
testY = to_categorical(testY)
# preview reshaped data
print(f'Train: X={trainX.shape}, Y={trainY.shape}')
print(f'Test: x={testX.shape}, y={testY.shape}')

#%% normalize
print(f'Center Slice Pre-Normalization:\n{trainX[0, 13:15, 13:15, 0]}')
# convert from integers to floats
trainX = trainX.astype('float32')
testX = testX.astype('float32')
#normalize to range 0-1
trainX = trainX / 255.0
testX = testX / 255.0
print(f'\nCenter Slice Post-Normalization:\n{trainX[0, 13:15, 13:15, 0]}')

#%% define cnn model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
# preview model
model.summary()
#visualizer(model, format='png', view=True)
# compile model
opt = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
