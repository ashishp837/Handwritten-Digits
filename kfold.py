import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.optimizers import SGD, Adam


# load data from keras
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

X_t = np.concatenate((X_train, X_test), axis=0)
print("new Concatenated train_images ", X_t.shape)

for i in range(70000):
    for j in range (784):
        if X_t[i][j]>=0.5:
            X_t[i][j]=1
        else:
            X_t[i][j]=0

X_t1 = X_t.reshape(X_t.shape[0], 28, 28)

y_t = np.concatenate((y_train, y_test), axis=0)
print("new Concatenated train_labels ", y_train.shape)

X_t, X_val, y_t, y_val = train_test_split(X_t, y_t, test_size = 0.2)
X_valnew = X_val

model = Sequential()
model.add(Dense(784, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
model.add(Dense(512, kernel_initializer='normal', activation='relu'))
model.add(Dense(256, kernel_initializer='normal', activation='relu'))
model.add(Dense(10, kernel_initializer='normal', activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer= 'Adam' , metrics=['accuracy'])

print(model.summary())
print('----------------------------------------------------------------------')

SEED=42
from sklearn import (metrics, cross_validation)
mean_auc = 0.0
n = 10  # repeat the CV procedure 10 times to get more precise results
for i in range(n):
    # for each iteration, randomly hold out 20% of the data as CV set
    X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
    X_t, y_t, test_size=0.2, random_state=i*SEED)
    y_plt=y_cv
    
    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_cv = X_cv / 255
    
    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_cv = np_utils.to_categorical(y_cv)

    print('Training model')
    history = model.fit(X_train, y_train, epochs=2, validation_data=(X_cv,y_cv), batch_size=800, verbose=2)
 
    preds = model.predict_classes(X_cv, verbose=0)

    # compute AUC metric for this CV fold
    fpr, tpr, thresholds = metrics.roc_curve(y_plt, preds,pos_label=2)
    fpr
    tpr
    roc_auc = metrics.auc(fpr, tpr)
    print ("AUC (fold %d/%d): %f" % (i + 1, n, roc_auc))
    mean_auc += roc_auc

print ("Mean AUC: %f" % (mean_auc/n)) 

X_val = X_val / 255
ytest1=model.predict_classes(X_val)

a1=[]
for i in range(14000):
    if ytest1[i] != y_val[i]:
        a1.append(i)

X_valnew = X_valnew.reshape(14000,28,28)

fig=plt.figure(figsize=(18,18))
columns = 5
rows = 5
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(X_valnew[a1[i-1]], cmap='gray', interpolation='none')
    plt.xlabel("Predicted Label: {}".format(ytest1[a1[i-1]]))
    plt.ylabel("True Label: {}".format(y_val[a1[i-1]]))
plt.show()


ytest2=model.predict(X_val)
ytest3=ytest1
ytest1=model.predict_classes(X_val)
def second_largest(numbers):
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1            
            else:
                m2 = x
    return m2 if count >= 2 else None

b=[]
for i in range(14000):
    if max(ytest2[i])-second_largest(ytest2[i])<0.1:
        b.append(i)

for i in range(len(b)):
    for j in range(10):
        if ytest2[b[i]][j]==second_largest(ytest2[b[i]]):
            ytest3[b[i]]=j

a1=[]
for i in range(14000):
    if ytest1[i] != y_val[i]:
        a1.append(i)
        
a2=[]
for i in range(14000):
    if ytest3[i] != y_val[i]:
        a2.append(i)
        
a3=[]
for i in range(14000):
    if ytest3[i] != ytest1[i]:
        a3.append(i)
        



len(a1)
len(a2)

import itertools
import numpy as np
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
cnf_matrix = confusion_matrix(ytest3, y_val)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=range(10))
plt.show()

a3=[]
for i in range(14000):
    if ytest1[i] == y_val[i]:
        a3.append(i)

x=range(10)
width = 0.3
vals_correct = [0]*10
vals_total = [0]*10

for i in range(10):
    for j in range(len(a3)):
        if ytest1[a3[j]]==i:
            vals_correct[i]=vals_correct[i]+1

for i in range(10):
    for j in range(14000):
        if ytest1[j]==i:
            vals_total[i]=vals_total[i]+1

vals_accuracy = [0]*10
for i in range(10):
    vals_accuracy[i]=vals_correct[i]/vals_total[i]*100
 
vals_error = [0]*10
for i in range(10):
    vals_error[i]=100-vals_accuracy[i]


%matplotlib inline
plt.style.use('ggplot')

x_pos = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

plt.figure(figsize=(10,7))
plt.bar(x, vals_error, color='green')
plt.xlabel("Class")
plt.ylabel("Error %")
plt.title("Error % for different classes")

plt.xticks(x, x_pos)

plt.show()