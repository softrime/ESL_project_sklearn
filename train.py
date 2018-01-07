import random
import numpy as np 
import codecs
from data_utils import load_CIFAR10
import matplotlib.pyplot as plt

def CV(X_train, y_train, fold=5, test=False):
  assert(X_train.shape[0]==y_train.shape[0]==50000)
  if test==True:
    X_train = X_train[:500]
    y_train = y_train[:500]
  X_train = np.reshape(X_train,(X_train.shape[0],-1))
  y_train = np.reshape(y_train,(y_train.shape[0],-1))
  x_batch_list = np.vsplit(X_train, 5)
  y_batch_list = np.vsplit(y_train, 5)
  y_batch_list = [y_batch_list[i].reshape(y_batch_list[i].shape[0], ) for i in range(len(y_batch_list))]
  return x_batch_list, y_batch_list


def do_LogisticRegression(x_train, y_train, x_test, C):
  from sklearn.linear_model import LogisticRegression
  lr = LogisticRegression(C=C) 
  lr.fit(x_train, y_train)
  y_test_pred = lr.predict(x_test)
  return y_test_pred

#def do_KNN(x_train, y_train, x_test):
  

def do_SVM(x_train, y_train, x_test):
  from sklearn.svm import SVC
  svm_model = SVC(probability=True, C=500)
  svm_model.fit(x_train, y_train)
  for i in range(x_test.shape[0]):
      #print i,x_test[i]
      y_test_pred = svm_model.predict([x_test[i]])
      #print y_test_pred;raw_input()
  return y_test_pred



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
# Cross Validation
x_train_folds, y_train_folds = CV(X_train, y_train, test=True)

C = 1.0
output_file = "result/LR_C-%f.result"% C
fin = codecs.open(output_file, 'w')
for i in range(5):
  print "Cross Validation: %d" % i

  x_train = np.array([0])
  y_train = np.array([0])
  for j in range(5):
    #print j;raw_input()
    if j == i:
      x_test = x_train_folds[j]
      y_test = y_train_folds[j]
    else:
      if x_train.sum() == 0 and y_train.sum() == 0:
        x_train = x_train_folds[j]
        y_train = y_train_folds[j]
      elif x_train.sum() != 0 and y_train.sum() != 0:
        x_train = np.concatenate((x_train, x_train_folds[j]), axis=0)
        y_train = np.concatenate((y_train, y_train_folds[j]), axis=0)
      else:
        print "error: x_train is None and y_train is not None or otherwise"
        exit(1)
  print ('x_train : ',x_train.shape, 'y_train : ', y_train.shape, 'x_test : ',x_test.shape);#raw_input()

  
  y_test_pred = do_LogisticRegression(x_train, y_train, x_test, C=C)
  num_correct = np.sum(y_test_pred == y_test)
  accuracy = float(num_correct) / len(y_test)
  fin.write('Got %d / %d correct => accuracy: %f \n' % (num_correct, len(y_test), accuracy))
fin.close()
#y_test_pred = do_SVM(x_train, y_train, x_test)



