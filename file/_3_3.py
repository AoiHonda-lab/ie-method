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

# from return_data import load_data
# from return_data_noise import load_data


def load_pkl(pklpath):
    with open(pklpath, mode="rb") as f:
        return pickle.load(f)

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

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):

    N, C, H, W = input_data.shape # N = num_data , C = channel
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def main():
	with open('./data/mnist/mnist.pkl', mode="rb") as f:
		train_all, test_all = pickle.load(f)
	with open('./data/mnist/mnist_2class.pkl', mode="rb") as f:
		train, test = pickle.load(f)
	print()
	# for i in range(9):
	# 	print("_num: {}".format(len(train[i])), end = "")
	# 	print("_num: {}".format(len(test[i])))
	# cnt=0
	image = []
	label = []
	train_all_3_3 = []
	for i in range(len(train_all)):
		x,t = train_all[i]
		x = x[np.newaxis, :, :]
		for i in range(3):
			if i == 2:
				x = F.average_pooling_2d(x, 2, stride =2)
			else:
				x = F.average_pooling_2d(x, 2)
		image.append(x[0][0].data)
		label.append(t)
	train_all_3_3.append(chainer.datasets.TupleDataset(image, label))	
	print()

	image = []
	label = []
	test_all_3_3 = []
	for i in range(len(test_all)):
		x,t = test_all[i]
		x = x[np.newaxis, :, :]
		for i in range(3):
			if i == 2:
				x = F.average_pooling_2d(x, 2, stride =2)
			else:
				x = F.average_pooling_2d(x, 2)
		image.append(x[0][0].data)
		label.append(t)
	test_all_3_3.append(chainer.datasets.TupleDataset(image, label))	
	print()

	pkl_data = []
	pkl_data.append(train_all_3_3)
	pkl_data.append(test_all_3_3)
	pickle.dump(pkl_data, open("mnist_data_3_3.pkl", "wb"), -1)
	return train, test



	

if __name__ == '__main__':
	main()
