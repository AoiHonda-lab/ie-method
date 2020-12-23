# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer import iterators, optimizers, initializers
import chainer.functions as F
import chainer.links as L
from chainer.datasets import split_dataset_random
from chainer.dataset import concat_examples
from matplotlib import pyplot
import pickle


#Titanicのエクセルデータをnumpyデータにします。
Titanicdata = np.genfromtxt("./data/create/artificial_data_v4.csv", delimiter=",").astype(np.float32)
datasize = Titanicdata.shape[0]-1

Y = Titanicdata[1:, 0].astype(np.float32)
X1 = Titanicdata[1:, 1:6]

# #chainerで扱いやすくするためにタプルにします。
X = chainer.datasets.TupleDataset(X1, Y)
# Tf = chainer.datasets.TupleDataset(T)

# #トレーニングデータの個数
# trainsize = args.bagging_counter


# #データをランダムにトレーニングデータ数と残りをテストデータに分ける
train, valid = split_dataset_random(X, 242)

# # train_box = []
# # test_box = []

# # train_box.append(train)
# # test_box.append(valid)

pkl_data = []
pkl_data.append(train)
pkl_data.append(valid)
#pickle.dump(pkl_data, open("./data/create/artificial.pkl", "wb"), -1)
pickle.dump(X, open("./data/create/artificial_all.pkl", "wb"), -1)
#pickle.dump(Tf, open("./data/Titanic/Titanic_test_1_class.pkl", "wb"), -1)

