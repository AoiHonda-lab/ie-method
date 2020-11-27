# -*- coding: utf-8; -*-
import os
import argparse
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
# from return_data import load_data
# from return_data_noise import load_data



def main():
    # 設定
    filename = "./result/train/ww/w_ww/shaplay/artificial_shapev4"
    # ファジィ測度のメビウス変換の値の変数の数
    syugou = 2
    # 変数の数
    omega = 5

    # mob_mob_fuzy= pd.read_csv("shape_ori.csv",  encoding="shift-jis")
    # mob_fuzy = mob_fuzy.values[0].tolist()
    mob_fuzy = pd.read_csv("{}.csv".format(filename),  encoding="shift-jis")
    mob_fuzy = mob_fuzy.values[0].tolist()

    
    # 集合の数取得関数
    # nCr
    def combinations_count(n, r):
        return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))

    # 部分集合取得関数
    def subsets(youso_num):
        subsets=[]
        for i in range(len(youso_num) + 1):
            for c in combinations(youso_num, i):
                # subsets.append(c)
                if not i == 0:
                    c = str(re.sub(",", "", "{}".format(c)))
                    c = re.sub(" ", "", "{}".format(c))
                    subsets.append(c)
                    
                else:
                    subsets.append(str(c).replace(")","0)"))
        return subsets
    
    num_syugou = 0
    add_syugou = 0
    # nC1 +nC2 + ...
    for i in range(1, omega + 1):
        num_syugou += combinations_count(omega,i)
    for i in range(1, syugou + 1):
        add_syugou += combinations_count(omega,i)
    
    for j in range(num_syugou - add_syugou):
        mob_fuzy.append(float(0))

    # 部分集合取得
    omega = list(range(1,omega+1))
    all_syugou = subsets(omega)
    # for i in range(len(omega) + 1):
    #     for c in combinations(omega, i):
    #         if not i == 0:
    #             c = str(re.sub(",", "", "{}".format(c)))
    #             c = re.sub(" ", "", "{}".format(c))
    #             all_syugou.append(c)
                
    #         else:
    #             all_syugou.append(str(c).replace(")","0)"))
    # all_syugou = all_syugou[0:len(omega)+1]




    # 集合とファジィ測度を辞書型に
    d_mob_pre = dict(zip(all_syugou, mob_fuzy))
    d_mob = dict(zip(all_syugou, mob_fuzy))
    # 辞書型をリスト型に
    list_d_mob = sorted(d_mob.items(), key=lambda x: x[0])
    # print(list_d_mob)
    
    # メビウス逆変換
    for ys in range(len(list_d_mob)):
        youso_num = []    
        num = re.sub("\\D", "", "{}".format(list_d_mob[ys][0]))
        for n in range(len(num)):
            youso_num.append(int(num[n]))
        subset = subsets(youso_num)
        fuzy = 0
        for l in range(1,len(subset)):
            fuzy += d_mob_pre["{}".format(subset[l])]
        d_mob["({})".format(num)] = fuzy
    
    """
    # 数字へ
    num = re.sub("\\D", "", "{}".format(list_d_mob[3][0]))
    # 部分集合を取るための要素取得
    for n in range(len(num)):
        youso_num.append(int(num[n]))
    subset = subsets(youso_num)
    all_fuzy = []
    fuzy = 0
    for l in range(1,len(subset)-1):
        fuzy += d_mob["{}".format(subset[l])]
    # fuzyの値取得
    print()
    
    # あとは辞書型で検索して置換
    d_mob["({})".format(num)] = fuzy
    

    # d_mob["{}".format(subset[0])]
    print()
    """




    l = list(d_mob)
    all_shape = []
    n = len(omega)
    for i in range(1, n+1):
        A = [s for s in l if "{}".format(i) not in s]
        sum_shape = 0
        for j in range(len(A)):
            a_str = str(re.sub("\\D", "", "{}".format(A[j])))
            if j == 0:
                A_num = int(a_str)
            else:
                A_num = len(a_str)
            key_a = A[j]
            # ??
            fuzy_a = d_mob["{}".format(key_a)]
            # 空集合対策　区集合->(0)にしているため
            if j == 0:
                a_i = [s for s in l[0:n+1] if "{}".format(i) in s]
                a_i = re.sub("\\D", "", "{}".format(a_i[0]))
            else:
                a_i = ''.join(sorted(a_str + str(i)))
                # a_i = [s for s in l if "{}".format(int(a_str)) in s and "{}".format(i) in s]
            # key_a_i = a_i[0]
            fuzy_a_i = d_mob["({})".format(a_i)]
            sum_shape += ( ( math.factorial(A_num) * math.factorial(n - A_num - 1) ) / (math.factorial(n)) ) * (fuzy_a_i - fuzy_a)
            # print( ( ( math.factorial(A_num) * math.factorial(n - A_num - 1) ) / (math.factorial(n)) ) * (fuzy_a_i - fuzy_a))
            # print(sum_shape)
            # print()
        
        all_shape.append(sum_shape)
    all_shape_ = np.array(all_shape)
    np.savetxt("{}_値.csv".format(filename), np.array([all_shape_,all_shape_/sum(all_shape_)]).T ,fmt='%.4f',delimiter=',')
    print(all_shape_)
    print()

    # sum_shape = 
        # l_in = [s for s in l if '1' not in s]
    # l_in.append(l[0])s
    # n = re.sub("\\D", "", "{}".format(l_in[4]))
    # print()

    # sum_shape = 0
    # a_str = str(re.sub("\\D", "", "{}".format(l_in[1])))
    # A = len(a_str)
    # n = len(omega)
    # key_a = l_in[1]
    # fuzy_a = d_shape["{}".format(key_a)]
    # a_i = [s for s in l if '2' in s and "1" in s]
    # key_a_i = a_i[0]
    # fuzy_a_i = d_shape["{}".format(key_a_i)]
    # d_a_in = re.sub("()", "", "{}".format(a_in_a[0]))
    # sum_shape =  ( ( math.factorial(A) * math.factorial(n - int(a_str) - 1) ) / (math.factorial(n)) ) * (fuzy_a_i - fuzy_a)

    
    
    

    



if __name__ == '__main__':
	main()
