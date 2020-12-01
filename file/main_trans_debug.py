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

# from return_data import load_data
from return_data import load_data

def main():
	# read config
	parser = argparse.ArgumentParser()
   
	parser.add_argument('--gpu_id', type=int, default=-1)
	parser.add_argument('--out', type=int, default=2, help='units_out in each layer')
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--epoch', type=int, default=100000, help='epoch for each generation')
	parser.add_argument('--func', type=str, default='relu', help='dataset: sigmoid or relu')
	parser.add_argument('--model', type=str, default='ie', help='dataset: mlp or cnn')
	parser.add_argument('--opt', type=str, default='sgd', help='dataset: sgd or adam')
	parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
	parser.add_argument('--limit', type=float, default=0.4, help='limit_train_data')
	parser.add_argument('--unit', type=float, default=800, help='learning rate')
	parser.add_argument('--data', type=str, default='mnist', help='dataset: car or mnist')
	parser.add_argument('--w_name', type=str, default='w_default', help='data_file_name')
	parser.add_argument('--data_model', type=str, default='_3_3_mnist_2class', help='data_file_name')


	args = parser.parse_args()


	# read data
	
	get_epoch = []
	for num in range(6):
		print("0_ " + str(num+1) + "")
		print()
		if args.model == "ie":
			ie_data = []
			for i in range(len(train[num])):

				ie_data.append(train[num][i][0].reshape(9))
			ie_data = np.array(ie_data)
		# data_iter

		# train_iter = iterators.SerialIterator(train[num][1:1000], args.batch_size)
		# test_iter = iterators.SerialIterator(test[num][1:1000], len(test[num]))

		train_iter = iterators.SerialIterator(train[num], args.batch_size)
		test_iter = iterators.SerialIterator(test[num], len(test[num]))
		
		# define model
		define_model = args.model
		if define_model == "mlp":
			model = mlp.MLP(args)
		elif define_model == "cnn":
			model = cnn.CNN(args)
		elif define_model == "ie":
			model = ie.IE(args, ie_data)

		
		print("===== model name =====", flush=True)
		print(define_model)
		print("===== model func =====", flush=True)
		print(args.func)
		print("===== model opt =====", flush=True)
		print(args.opt)
		print("===== model lr =====", flush=True)
		print(args.lr)

		# define optimizer
		optimizer = chainer.optimizers
		define_opt = args.opt
		if define_opt == "sgd":
			optimizer = optimizer.SGD(lr=args.lr)
		elif define_opt == "adam":
			optimizer = optimizer.Adam(alpha=args.lr)
		optimizer.setup(model)


		# train start
		elapsed_time = []
		start = time.time()

		
		train_loss, test_loss, test_last_loss, train_acc, test_acc, test_last_acc, epoch = model.train_model(train_iter, test_iter, optimizer, args)

		elapsed_time.append(time.time() - start)
		print("===== read time =====", flush=True)
		print('time:{}'.format(elapsed_time[0]))
		# train_loss, test_loss, w_b = model.train_model(train_iter, test_iter, optimizer)
				
		out_loss = [["epoch",'train_loss','test_loss', 'train_acc', 'test_acc']]
		for idx, (train_loss, test_loss, train_acc, test_acc ) in enumerate(zip(train_loss, test_loss, train_acc, test_acc)):
			out_loss.append([idx+1,train_loss, test_loss, train_acc, test_acc])
		out_loss.append(test_last_loss)
		out_loss.append(test_last_acc)
		out_loss.append([elapsed_time[0]])

		with open('./result/value/result_{}_{}_{}_{}_{}_{}_{}_{}_0_{}.csv'.format(args.day, args.add, args.func, args.opt, args.lr, args.ie_op, args.data_model, args.model, num+1), 'w') as f:
			writer = csv.writer(f, lineterminator='\n', delimiter='\t')
			writer.writerows(out_loss)
		
		# CPU環境でも学習済みモデルを読み込めるようにCPUに移してからダンプ
		model.to_cpu()
		pickle.dump(model, open("./result/pkl/model_{}_{}_{}_{}_{}_{}_{}_{}_0_{}.pkl".format(args.day, args.add, args.func, args.opt, args.lr ,args.ie_op, args.data_model, args.model, num+1), "wb"), -1)

		get_epoch.append(epoch)
		
		print()
	
	out_epoch = [["0_1", "0_2", "0_3", "0_4","0_5", "0_6", "0_7", "0_8", "0_9"]]
	out_epoch.append([get_epoch[0], get_epoch[1], get_epoch[2], get_epoch[3], get_epoch[4], get_epoch[5], get_epoch[6], get_epoch[7], get_epoch[8]])
	out_epoch = [["0_1", "0_2", "0_3", "0_4", "0_5"]]
	out_epoch.append([get_epoch[0], get_epoch[1], get_epoch[2], get_epoch[3], get_epoch[4]])
	with open('./result/value/{}_{}_{}_{}_{}_{}_{}_{}_epoch.csv'.format(args.day, args.add, args.func ,args.opt, args.lr , args.ie_op, args.data_model, args.model), 'w') as f:
		writer = csv.writer(f, lineterminator='\n', delimiter='\t')
		writer.writerows(out_epoch)
	


	# read data
	
	print("===== read data =====", flush=True)
	print()
	with open('./data/mnist/{}.pkl'.format(args.data_model), mode="rb") as f:
		train, test = pickle.load(f)
	
	max_epoch = []
	

	for pre in range(1,10):
		max_epoch.append([])
		get_epoch = []
		for post in range(1,10):
			
			# 重複転移
			if pre == post:
				get_epoch.append(0)
				continue
			
			print("0_{} --> 0_{}".format(pre, post))
			print()

			# # データの例示
			# for i in range(100):
			# 	if train[post-1][i][1] == 1:
			# 		x,t = train[post-1][i]
			# plt.imshow(x.reshape(28, 28), cmap='gray')
			# plt.axis('off')
			# # plt.show()
			# print('label:', t)

			# data_iter
			# args.batch_size = len(train)
			train_iter = iterators.SerialIterator(train[post-1], args.batch_size)
			test_iter = iterators.SerialIterator(test[post-1], len(test[post-1]))

			# get pre_model
			with open('./result/pkl/model_{}_{}_0_{}.pkl'.format(args.data_model,args.model, pre), mode="rb") as f:
				pre_model = pickle.load(f)
			
			# define model
			model = pre_model

			# define optimizer
			optimizer = chainer.optimizers
			define_opt = args.opt
			if define_opt == "sgd":
				optimizer = optimizer.SGD(lr=args.lr)
			elif define_opt == "adam":
				optimizer = optimizer.Adam(alpha=args.lr)
			optimizer.setup(model)

			# train start
			elapsed_time = []
			start = time.time()

			# train and test
			train_loss, test_loss, test_last_loss, train_acc, test_acc, test_last_acc, epoch = model.train_model(train_iter, test_iter, optimizer, args)

			elapsed_time.append(time.time() - start)
			print("===== read time =====", flush=True)
			print('time:{}'.format(elapsed_time[0]))


			# get result data 
			out_loss = [["epoch",'train_loss','test_loss', 'train_acc', 'test_acc']]
			for idx, (train_loss, test_loss, train_acc, test_acc ) in enumerate(zip(train_loss, test_loss, train_acc, test_acc)):
				out_loss.append([idx+1,train_loss, test_loss, train_acc, test_acc])
			out_loss.append(test_last_loss)
			out_loss.append(test_last_acc)
			out_loss.append([elapsed_time[0]])
			with open('./result/value/test_transfer_{}_{}_0_{}__0_{}.csv'.format(args.data_model, args.model, pre, post), 'w') as f:
				writer = csv.writer(f, lineterminator='\n', delimiter='\t')
				writer.writerows(out_loss)

			# get epoch
			get_epoch.append(epoch)
			



			# CPU環境でも学習済みモデルを読み込めるようにCPUに移してからダンプ
			model.to_cpu()
			pickle.dump(model, open("./result/pkl/test_trans_{}_{}_0_{}__0_{}.pkl".format(args.data_model, args.model, pre, post), "wb"), -1)
		
		
		max_epoch[-1].append(get_epoch)
		print()
	

	out_epoch = [["pre", "0_1", "0_2", "0_3", "0_4","0_5", "0_6", "0_7", "0_8", "0_9"]]
	for num, max_epoch in enumerate(zip(max_epoch)):
		out_epoch.append(["0_{}".format(num+1), max_epoch[0][0][0], max_epoch[0][0][1], max_epoch[0][0][2], max_epoch[0][0][3], max_epoch[0][0][4], max_epoch[0][0][5], max_epoch[0][0][6], max_epoch[0][0][7], max_epoch[0][0][8]])

	with open('./result/value/test_transfer_{}_{}.csv'.format(args.data_model, args.model), 'w') as f:
		writer = csv.writer(f, lineterminator='\n', delimiter='\t')
		writer.writerows(out_epoch)





	

if __name__ == '__main__':
	main()