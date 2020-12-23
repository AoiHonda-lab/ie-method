# -*- coding: utf-8; -*-
import os
import argparse
import random
import pickle
import sys
import copy

import mlp
import cnn
import ie


import pandas as pd
import numpy as np
import csv
import time
import pylab
import matplotlib.pyplot as plt


import chainer
from chainer import iterators
from chainer import optimizers
from chainer import serializers
from chainer.datasets import split_dataset_random

#タイタニックテストデータ予測
with open('./result/train/pkl/model_9_10_1_mse_2_sigmoid_MomentumSGD_0.001_False_Titanic_all_1_class_monotony_ie_False.pkl', mode='rb') as f:
	model = pickle.load(f)
with open('./data/Titanic/{}.pkl'.format("Titanic_test_1_class"), mode="rb") as f:
	Rtest = pickle.load(f)
test_suvive = []
test_ID = []
for i in range(len(model(Rtest._datasets[0]))):
    test_ID.append(int(892+i))
    if model(Rtest._datasets[0])[i].data[0] > 1/2:
        test_suvive.append(1)
    else:
        test_suvive.append(0)
np.savetxt('./result/test/{}_{}_{}_{}.csv'.format("9_10","Titanic_test_result","Momentum","5000"),np.array([test_ID,test_suvive],dtype = 'int32').T, header='PassengerId,Survived',fmt='%d',delimiter=',',comments='')