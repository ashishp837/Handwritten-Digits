import numpy as np
import pandas as pd
import pydot
import seaborn as sns
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD, Adam


# load data from keras
(X_train, y_train), (X_test, y_test) = mnist.load_data()

a=[]
for i in range(91):
    if y_train[i] == 4:
        a.append(i)

fig = plt.figure()
for i in range(10):
  plt.subplot(2,5,i+1)
  plt.tight_layout()
  plt.imshow(X_train1[a[i]], cmap='gray', interpolation='none')
  plt.title("Actual Label: {}".format(y_train[a[i]]))
  plt.xticks([])
  plt.yticks([])

# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

#CONCATENATE the training sets of Kaggle and Keras into final TRAIN

X_train = np.concatenate((X_train, X_test), axis=0)
print("new Concatenated train_images ", X_train.shape)

y_train = np.concatenate((y_train, y_test), axis=0)
print("new Concatenated train_labels ", y_train.shape)

g = sns.countplot(y_train)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2)

y_plt=y_val

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
num_classes=10

epochs=12

model = Sequential()
model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
model.add(Dense(784, kernel_initializer='normal', activation='relu'))
model.add(Dense(784, kernel_initializer='normal', activation='relu'))
model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer= 'Adam' , metrics=['accuracy'])

print(model.summary())
print('----------------------------------------------------------------------')
print('Training model')
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val,y_val), batch_size=800, verbose=1)

plt.rcParams["figure.figsize"]=(20,10)
for key in history.history.keys():
    plt.plot(range(1,epochs+1),history.history[key])

plt.legend(list(history.history.keys()),loc='upper left')
plt.title('no idea')
plt.show()

y_test = model.predict_classes(X_val, verbose=2)

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_plt)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=range(10))
plt.show()

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




fun(1,1)

from mpl_toolkits import mplot3d
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import math

def fun(x,y):
    a=cnf_matrix[math.floor(x),math.floor(y)]
    return a;

def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

x = np.linspace(0, 9, 50)
y = np.linspace(0, 9, 50)

X, Y = np.meshgrid(x, y)
for i in range(50):
    Z[i]=fun(x[i],y[i])
Z = f(X, Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')