#%% load mnist dataset
from keras.datasets import mnist
from tensorflow.keras import utils as np_utils
from matplotlib import pyplot
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
trainY = np_utils.to_categorical(trainY)
testY = np_utils.to_categorical(testY)
# preview reshaped data
print(f'Train: X={trainX.shape}, Y={trainY.shape}')
print(f'Test: x={testX.shape}, y={testY.shape}')
