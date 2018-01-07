import random
import numpy as np 
from data_utils import load_CIFAR10
import matplotlib.pyplot as plt





# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.

#%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#%load_ext autoreload
#%autoreload 2

cifar10_dir = '../cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# subsample data for more dfficient code execution 
#num_training = 500
#range(5)=[0,1,2,3,4]
#mask = range(num_training)
#x_train = X_train[mask]
#y_train = y_train[mask]
#num_test = 100
#mask = range(num_test)
#x_test = X_test[mask]
#y_test = y_test[mask]
#the image data has three chanels
#the next two step shape the image size 32*32*3 to 3072*1
#x_train = np.reshape(x_train,(x_train.shape[0],-1))
#x_test = np.reshape(x_test,(x_test.shape[0],-1))
x_train = np.reshape(X_train,(X_train.shape[0],-1))
x_test = np.reshape(X_test,(X_test.shape[0],-1))
print ("after subsample and re shape:")
print ('x_train : ',x_train.shape, 'y_train : ', y_train.shape, 'x_test : ',x_test.shape)

def do_LogisticRegression(x_train, y_train, x_test, C):
  from sklearn.linear_model import LogisticRegression
  lr = LogisticRegression(C=C) 
  lr.fit(x_train, y_train)
  y_test_pred = lr.predict(x_test)
  return y_test_pred

def do_KNN(x_train, y_train, x_test):
  

def do_SVM(x_train, y_train, x_test):
  from sklearn.svm import SVC
  svm_model = SVC(probability=True, C=500)
  svm_model.fit(x_train, y_train)
  for i in range(x_test.shape[0]):
      print i,x_test[i]
      y_test_pred = svm_model.predict([x_test[i]])
      print y_test_pred;raw_input()
  return y_test_pred

C = 1.0
y_test_pred = do_LogisticRegression(x_train, y_train, x_test, C=C)
#y_test_pred = do_SVM(x_train, y_train, x_test)

# Compute and print the fraction of correctly predicted examples
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / len(y_test)

output_file = "result/LR_C-%f.result"% C
fin = codecs.open(output_file, 'w')
fin.write('Got %d / %d correct => accuracy: %f' % (num_correct, len(y_test), accuracy))
fin.close()
