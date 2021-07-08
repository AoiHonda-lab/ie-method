# -*- coding: utf-8; -*-
import os
import argparse
from typing import ItemsView
import numpy as np
import random
import pickle
import sys
import copy
import chainer
from chainer import iterators
import chainer
import chainer.links as L
import chainer.functions as F
import matplotlib.pyplot as plt
import csv
import math
import time
from chainer import optimizers
from chainer import serializers
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
import re
import itertools
import calc
# from return_data import load_data
# from return_data_noise import load_data



#w:重み syugou:args.add omega:変数の数
def get_shape(w, syugou, omega, args):
    shape_box = []
    # omega = 9
    mob_fuzy = w

    def daisu(ie_data_len, args):
    # 代数積を取得
        items = [i for i in range(1, ie_data_len+1)]
        subsets=[]
        for i in range(len(items) + 1):
            if i > args.add:#二加法的まで
                break
            for c in combinations(items, i):
                subsets.append(c)
                # subsets.append(list(c))
        hh = subsets
        return hh 

    # 部分集合取得
    if args.matrixtype == 2:
        all_syugou = calc.rnn_matrix_tuple(omega)
    elif args.matrixtype == 3:
        all_syugou = calc.bi_rnn_matrix_tuple(omega)
    else:
        all_syugou = daisu(omega, args)
    d_mob = dict(zip(all_syugou, mob_fuzy)) #辞書化して各々対応した要素に重みを入れている
    l = list(d_mob) #list化した空集合を含む集合
    shap_sum_i = []
    for i in range(1, len(l)):
        # key_a = l[j]
        shap = []
        if i == 1:
            search = 1
        elif len(l[i])-len(l[i-1]) == 1:
            search = i
        else:
            pass
        for j in range(search, len(l)):
            T = l[i]
            length = len(l[j])
            if  set(T) <= set(l[j]):
                w = 1/(len(l[j])-len(T)+1) * d_mob[l[j]]
                shap.append(w)
        shap_sum_i.append(sum(shap))
        

    return shap_sum_i