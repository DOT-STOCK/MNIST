#%% 
# load mnist dataset
from keras.datasets import mnist
from matplotlib import pyplot

#%% 
# preview data
(trainX, trainY), (testX, testY) = mnist.load_data()
print(f'Train: X={trainX.shape}, Y={trainY.shape}')
print(f'Test: x={testX.shape}, y={testY.shape}')

#%% 
# plot first few images
for i in range(9):
    pyplot.subplot(331 + i)
    pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()