import numpy as np
import chainer
from chainer import iterators, optimizers, initializers
import chainer.functions as F
import chainer.links as L
from chainer.datasets import split_dataset_random
from chainer.dataset import concat_examples
from matplotlib import pyplot
import pickle

#def train_valid(args,num):
#Titanicのエクセルデータをnumpyデータにします。
Titanicdata = np.genfromtxt("./data/Titanic/Titanic_train_v2.csv", delimiter=",", filling_values=0).astype(np.float32)
testdata = np.genfromtxt("./data/Titanic/Titanic_test2.csv", delimiter=",", filling_values=0).astype(np.float32)
testsize = testdata.shape[0]-1
datasize = Titanicdata.shape[0]-1

T = testdata[1:testsize+1, 0:8]

Y = Titanicdata[1:datasize+1, 0].astype(np.int32)
X1 = Titanicdata[1:datasize+1, 1:9]

#chainerで扱いやすくするためにタプルにします。
X = chainer.datasets.TupleDataset(X1, Y)
Tf = chainer.datasets.TupleDataset(T)

#トレーニングデータの個数
trainsize = 100


#データをランダムにトレーニングデータ数と残りをテストデータに分ける
train, valid = split_dataset_random(X, trainsize)

train_box = []
test_box = []

train_box.append(train)
test_box.append(valid)

pkl_data = []
pkl_data.append(train)
pkl_data.append(valid)
#pickle.dump(pkl_data, open("./data/Titanic/{}Titanic_normal_1_class.pkl".format(str(num)), "wb"), -1)
pickle.dump(X, open("./data/Titanic/Titanic_v2_1_class.pkl", "wb"), -1)
#pickle.dump(Tf, open("./data/Titanic/Titanic_test_1_class.pkl", "wb"), -1)

