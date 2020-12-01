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
import time
from chainer import optimizers
from chainer import serializers
import sklearn.decomposition as decomp
# from return_data import load_data
# from return_data_noise import load_data
# function definitions
class DigitData:
    def __init__(self, data):
        self.label = data[0]
        self.data  = data[1:785]
    def getLabel(self):
        return self.label
    def getData(self):
        return self.data
    def __repr__(self):
        return "label: " + str(self.label) + "\ndata: " + str(self.data) + "\n"

class DigitDataSet:
    def __init__(self, data):
        self.dataset = {}
        self.data = data[:,1:785]
        self.label = data[:,0]
        for d in data:
            item = DigitData(d)
            if item.getLabel() not in self.dataset:
                self.dataset[item.getLabel()] = [item.getData()]
            else:
                self.dataset[item.getLabel()].append(item.getData())

    def getLabel(self):
        return self.label
    
    def getData(self, index=-1):
        if index==-1:
            return self.data
        else:
            return self.data[index]
                
    def getByLabel(self, label, num=None):
        if label < 0 or 9 < label:
            raise Exception('num should be from 0 to 9.')
            
        if num is None:
            return np.array(self.dataset[label][0])
        if isinstance(num, int) or isinstance(num, float):
            return np.array(self.dataset[label][0:num])
        if num == 'all':
            return np.array(self.dataset[label])
        else:
            raise Exception('num should be int or float.')
    
    def __repr__(self):
        ret_val = ""
        for k in self.dataset.keys():
            ret_val += str(k) + ", " + str(len(self.dataset[k])) +"\n"
        return ret_val

def draw_digit(data, height, width, cnt, number):
    plt.subplot(2, 2, cnt+1)
    Z = data.reshape(28, 28)
    Z = Z[::-1]
    plt.xlim(0, 28 - 1)
    plt.ylim(0, 28 - 1)
    plt.pcolor(Z)
    plt.title("labels = {0}".format(number), size = 8)
    plt.gray()
    plt.tick_params(labelbottom = "off")
    plt.tick_params(labelleft = "off")


def main():
	with open('./data/mnist/pcd_train_mnist_1class.pkl', mode="rb") as f:
		train= pickle.load(f)
	with open('./data/mnist/pcd_test_mnist_1class.pkl', mode="rb") as f:
		test = pickle.load(f)
	print()
	# データの読み込み


	# raw_data= np.loadtxt('./data/mnist/train.csv',delimiter=',',skiprows=1)
	#dataset = DigitDataSet(raw_data)
	# dataset = DigitDataSet(raw_data)
	# data = [None for i in range(10)]
	# for i in range(10):
	# 	data[i] = dataset.getByLabel(i,'all')

	# 主成分分析の実行
	# 削減後の次元数による寄与率の違いを算出
	comp_items = [9]  # 削減後次元数のリスト
	cumsum_explained = np.zeros((10,len(comp_items)))
	pcd_train = []
	for i, n_comp in zip(range(len(comp_items)), comp_items):
		for num in range(10):                        # 各数字ごとに分析をかける
			pca = decomp.PCA(n_components = n_comp)  # 主成分分析オブジェクトの作成
			pca.fit(train[0][num])                       # 主成分分析の実行
			transformed = pca.transform(train[0][num])   # データに対して削減後のベクトルを生成
			pcd_train.append(transformed)
			E = pca.explained_variance_ratio_        # 寄与率
			cumsum_explained[num, i] = np.cumsum(E)[::-1][0] # 累積寄与率

	print ("|　label　|explained n_comp:5|explained n_comp:10|explained n_comp:20|explained n_comp:30|")
	print ("|:-----:|:-----:|:-----:|:-----:|:-----:|")
	for i in range(10):
		print ("|%d|%.1f％|"%(i, cumsum_explained[i,0]*100))
	print()
	v = pcd_train[0].tolist
	print()


	pcd_train_dataset = []	
	pcd_train_dataset_all = []	
	

	for num in range(10):
		image = []
		label = []
		for length in range(len(pcd_train[num])):
			image.append(pcd_train[num][length].reshape(3,3))
			label.append(num)
		pcd_train_dataset.append(chainer.datasets.TupleDataset(image, label))
	
	image = []
	label = []
	for num in range(10):
		for length in range(len(pcd_train[num])):
			image.append(pcd_train[num][length].reshape(3,3))
			label.append(num)

	pcd_train_dataset_all.append(chainer.datasets.TupleDataset(image, label))
	



	####test
	comp_items = [9]  # 削減後次元数のリスト
	cumsum_explained = np.zeros((10,len(comp_items)))
	pcd_test = []
	for i, n_comp in zip(range(len(comp_items)), comp_items):
		for num in range(10):                        # 各数字ごとに分析をかける
			pca = decomp.PCA(n_components = n_comp)  # 主成分分析オブジェクトの作成
			pca.fit(test[0][num])                       # 主成分分析の実行
			transformed = pca.transform(test[0][num])   # データに対して削減後のベクトルを生成
			pcd_test.append(transformed)
			E = pca.explained_variance_ratio_        # 寄与率
			cumsum_explained[num, i] = np.cumsum(E)[::-1][0] # 累積寄与率

	print ("|　label　|explained n_comp:5|explained n_comp:10|explained n_comp:20|explained n_comp:30|")
	print ("|:-----:|:-----:|:-----:|:-----:|:-----:|")
	for i in range(10):
		print ("|%d|%.1f％|"%(i, cumsum_explained[i,0]*100))
	print()
	v = pcd_test[0].tolist
	print()


	pcd_test_dataset = []	
	pcd_test_dataset_all = []	
	

	for num in range(10):
		image = []
		label = []
		for length in range(len(pcd_test[num])):
			image.append(pcd_test[num][length].reshape(3,3))
			label.append(num)
		pcd_test_dataset.append(chainer.datasets.TupleDataset(image, label))
	
	image = []
	label = []
	for num in range(10):
		for length in range(len(pcd_test[num])):
			image.append(pcd_test[num][length].reshape(3,3))
			label.append(num)

	pcd_test_dataset_all.append(chainer.datasets.TupleDataset(image, label))
	print()

	pkl_data = []
	pkl_data.append(pcd_train_dataset)
	pkl_data.append(pcd_test_dataset)
	pickle.dump(pkl_data, open("pcd_mnist_data.pkl", "wb"), -1)

	pkl_data_all = []
	pkl_data_all.append(pcd_train_dataset_all)
	pkl_data_all.append(pcd_test_dataset_all)
	pickle.dump(pkl_data_all, open("pcd_mnist_data_all.pkl", "wb"), -1)
		



    


if __name__ == '__main__':
	main()
