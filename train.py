import sys

import random
import numpy as np 
import codecs
from data_utils import load_CIFAR10
from data_utils import load_CIFAR_batch
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
from scipy import misc

def CV(X_train, y_train, fold=5, test=False):
  assert(X_train.shape[0]==y_train.shape[0]==50000)
  if test==True:
    X_train = X_train[:1000]
    y_train = y_train[:1000]
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

def do_KNN(x_train, y_train, x_test, n_neighbors=5):
  from sklearn.neighbors import KNeighborsClassifier
  knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
  knn_model.fit(x_train, y_train)
  y_test_pred = knn_model.predict(x_test)
  #print y_test_pred, y_test_pred.shape;raw_input()
  return y_test_pred

  

def do_SVM(x_train, y_train, x_test, C=500, kernel='rbf'):
  from sklearn.svm import SVC
  svm_model = SVC(probability=True, C=C, kernel=kernel)
  svm_model.fit(x_train, y_train)
  for i in range(x_test.shape[0]):
      #print i,x_test[i]
      y_test_pred = svm_model.predict([x_test[i]])
      #print y_test_pred;raw_input()
  return y_test_pred

def load_test_data():
  scp_file = 'test.scp'
  fin = codecs.open(scp_file, 'r');images = fin.readlines();fin.close()
  test_data = []
  for i in range(len(images)):
    basename = images[i].strip().split('\n')[0]
    path = '../testData/' + basename
    im =plb.imread(path)
    im_reshape = misc.imresize(im, (32, 32)) 
    test_data.append(im_reshape)
  test_data = np.array(test_data)
  return test_data
    
    


def train():
  if len(sys.argv) < 3:
    print "need at least 2 parameters"
    exit(1)

  model = sys.argv[1] 
  if len(sys.argv) == 4:
    param1 = sys.argv[2]
    param2 = sys.argv[3]
  else:
    param1 = sys.argv[2]


  if len(sys.argv) == 4:
    print "========== Model: %s | PARAM1: %s | PARAM2: %s ==========" % (model, str(param1), str(param2))
  else:
    print "========== Model: %s | PARAM1: %s ==========" % (model, str(param1))

  cifar10_dir = '../cifar-10-batches-py'
  X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
  # Cross Validation
  x_train_folds, y_train_folds = CV(X_train, y_train, test=True)

  if model == 'SVM':
    output_file = "result/SVM_C-%f_kernel-%s.result"% (float(param2), param1)
  elif model == 'KNN':
    output_file = "result/KNN_k-%f.result"% float(param1)
  elif model == 'LR':
    output_file = "result/LR_C-%f.result"% float(param1)
  else:
    print "unexpected model : %s" % model
    exit(1)

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

    if model == 'LR':
      y_test_pred = do_LogisticRegression(x_train, y_train, x_test, C=float(param1))
    elif model == 'KNN':
      y_test_pred = do_KNN(x_train, y_train, x_test, int(param1))
    elif model == 'SVM':
      y_test_pred = do_SVM(x_train, y_train, x_test, float(param2), kernel=param1)
    else:
      print "unexpected model : %s" % model
      exit(1)

    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / len(y_test)
    fin.write('Got %d / %d correct => accuracy: %f \n' % (num_correct, len(y_test), accuracy))
  fin.close()

def knn_predict():
  if len(sys.argv) < 3:
    print "need at least 2 parameters"
    exit(1)

  model = sys.argv[1] 
  param1 = sys.argv[2]

  cifar10_dir = '../cifar-10-batches-py'
  X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
  #X_train, y_train = load_CIFAR_batch(cifar10_dir + '/data_batch_1')

  X_train = X_train[:1000]
  y_train = y_train[:1000]
  test_data = load_test_data()

  X_train = np.reshape(X_train,(X_train.shape[0],-1))
  y_train = np.reshape(y_train,(y_train.shape[0],-1))
  test_data = np.reshape(test_data, (test_data.shape[0], -1))

  y_pred = do_LogisticRegression(X_train, y_train, test_data, C=66)
  
  print y_pred.shape

  scp_file = 'test.scp'
  fin = codecs.open(scp_file, 'r');images = fin.readlines();fin.close()

  assert(len(images)==y_pred.shape[0])
  output = codecs.open('prediction.txt', 'w')
  for i in range(len(images)):
    basename = images[i].split('\n')[0]
    output.write(basename + ' ' + str(y_pred[i]) + '\n')
  output.close()
  



if __name__=='__main__':
  #train()
  knn_predict()
