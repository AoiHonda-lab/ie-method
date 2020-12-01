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


# 画像が反転してしまう
def draw_digit(data, height, width, cnt, number):
    plt.subplot(2, 2, cnt+1)
    # Z = data.reshape(height, width)
    Z = data
    plt.xlim(0, width )
    plt.ylim(0, height)
    plt.pcolor(Z.data)
    plt.title("size = {0} * {0}".format(number), size = 8)
    plt.gray()
    plt.tick_params(labelbottom = "off")
    plt.tick_params(labelleft = "off")


# # 画像を3回畳み込み
# def conv_sample(img):
#     titles = ["Original", "1st ", "2nd ", "3rd "]
#     fig = plt.figure(figsize = (8, 8))
#     fig.subplots_adjust(hspace=0.2, wspace=0.2)
#     for i in range(4):
#         ax = fig.add_subplot(2, 2, i+1)
#         view_img = img / np.max(img) #表示用にスケール調整
#         ax.imshow(view_img)
#         ax.set_title(titles[i] + " " + str(view_img.shape))

# # 画像を1回畳み込み
# def F_1(img):
#     titles = ["before", "after "]
#     fig = plt.figure(figsize = (8, 8))
#     fig.subplots_adjust(hspace=0.2, wspace=0.2)
#     for i in range(4):
#         ax = fig.add_subplot(2, 1, i+1)
#         view_img = img / np.max(img) #表示用にスケール調整
#         ax.imshow(view_img)
#         ax.set_title(titles[i] + " " + str(view_img.shape))

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
	x,t = test_all[13]
	# plt.imshow(x.reshape(28, 28), cmap='gray')
	# plt.axis('off')
	# plt.show()
	# draw_digit(x[0][0], h, w,  cnt, h)
	# plt.savefig("1.png")

	print(t)
	x = x[np.newaxis, :, :]
	cnt = 0
	# h, w = x[0][0].shape
	# cnt += 1

	# draw_digit(x[0][0], h, w,  cnt, h)
	for i in range(2):
		if i == 0:
			x = x
		elif i == 1:
			x = F.average_pooling_2d(x, 10, stride = 9, pad = 0)
		h, w = x[0][0].shape
		if i ==0:

			plt.imshow(x.reshape(28, 28), cmap='gray')
			plt.axis('off')
			plt.savefig("{}_before.png".format(t))
		if i == 1:
			plt.imshow(x.data[0][0], cmap='gray')
			plt.axis('off')
			plt.savefig("{}_after.png".format(t))
			# draw_digit(x[0][0], h, w,  cnt, h)
		# cnt += 1

		
		



	

if __name__ == '__main__':
	main()
