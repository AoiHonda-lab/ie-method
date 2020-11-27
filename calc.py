# -*- coding: utf-8; -*-
import os
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import pprint
import random
import chainer
from sklearn.model_selection import KFold
from itertools import chain,combinations
from chainer.optimizer_hooks import WeightDecay, Lasso
import chainer.functions as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve,roc_auc_score, precision_recall_fscore_support, r2_score, mean_squared_error
# n C r
def combinations_count(n, r):
	return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))

def daisu(ie_data_len, args):
# 代数積を取得
	items = [i for i in range(1, ie_data_len+1)]
	subsets=[]
	for i in range(len(items) + 1):
		if i > args.add:#二加法的まで
			break
		for c in combinations(items, i):
			subsets.append(list(c))
	hh = subsets
	return hh 

def add(arg_add, ie_data_len):
	add = 0
	for i in range(1,arg_add + 1):
		add += combinations_count(ie_data_len,i)
	return add


def set_sum(hh):
# 各点集合の長さ[9, 45, 129, ,,, 510, 511]
	set_sum = []
	for num in range(0, len(hh)):
		if (len(hh)-1==num):
			set_sum.append(num)
		elif (len(hh[num +1])) > len(hh[num]):
			set_sum.append(num)
		else:
			pass
	return set_sum

def siki(add, ie_data_len):
# 中間層ー出力層の初期値取得
	siki = []
	for i in range(1, add +1):
		if i <= ie_data_len:
			siki.append([1/ie_data_len])
		else:
			siki.append([0])
	return siki

def each_ie_data_and_cov(train):
	ie_data = []
	for each in range(len(train)):
		ie_data.append([])
		for i in range(len(train[0])):
			ie_data[each].append(train[each][i][0])
		ie_data[each] = np.array(ie_data[each])

	# 相関係数の値取得
	cov = []
	cov_data = []
	for each in range(len(train)):
		cov_data.append([])
		for i in range(len(train[-1])):
			cov_data[each].append(np.append(train[each][i][0].reshape(len(train[-1][-1][0])), train[each][i][1]))
		cov.append(np.corrcoef(np.array(cov_data[each]).T)[-1])
	
	return ie_data, cov

def ie_data_and_cov(train):
	ie_data = []
	# h, v = train[-1][0].shape
	for i in range(len(train)):
		# ie_data.append(train[i][0].reshape(h*v))
		ie_data.append(train[i][0])
	ie_data = np.array(ie_data)
	cov_data = []
	for i in range(len(train)):
		cov_data.append(np.append(train[i][0].reshape(len(ie_data[0])), train[i][1]))
	cov = np.corrcoef(np.array(cov_data).T)[len(ie_data[0])]

	return ie_data, cov

# 単調性の設定
def monotony(self, train_iter, args):
	#単調性の条件を緩くする変数を作成
	mono = args.mono
	for i in range(self.set_sum[self.args.add]):
		# 長さ＝○○点集合
		length = len(self.hh[1:][i])
		sum = 0
		for num in self.hh[1:][i]:
			for k in range(self.set_sum[length -1]):
				if num in self.hh[1:][k] == "True":
					sum += self.lt.W.data[0, k]
				else:
					pass
			if self.lt.W.data[0, i] + mono / (train_iter.epoch + 1) < -sum:
				self.lt.W.data[0, i] = -sum-mono / (train_iter.epoch + 1)
			else:
				pass
	

# 単調性の設定
def monotony_each(self, train_iter_s, ie_data_len, args):
	#単調性の条件を緩くする変数を作成
	mono = args.mono
	for each in range(ie_data_len):
		for i in range(self.set_sum[self.args.add]):
			# 長さ＝○○点集合
			length = len(self.hh[1:][i])
			sum = 0
			for num in self.hh[1:][i]:
				for k in range(self.set_sum[length -1]):
					if num in self.hh[1:][k] == "True":
						sum += self.ie_model[each][-1].W.data[0, k]
					else:
						pass
				if self.ie_model[each][-1].W.data[0, i] + mono / (train_iter_s[-1].epoch + 1) < -sum:
					self.ie_model[each][-1].W.data[0, i] = -sum-mono / (train_iter_s[-1].epoch + 1)
				else:
					pass

def norm(model, args):
	# setting norm 
	if args.norm == "l1":
		for param in model.params():
			if param.name != 'b' or len(param.shape):  # バイアス以外だったら
				# l1_norm
				param.update_rule.add_hook(WeightDecay(args.l_lr))  # 重み減衰を適用
	elif args.norm == "l2":
		for param in model.params():
			if param.name != 'b' or len(param.shape):  # バイアス以外だったら
				# l2_norm
				param.update_rule.add_hook(Lasso(args.l_lr))  # 重み減衰を適用
	else:
		pass

def craft_titanic(model, args, data, no):
	test_ID = []
	for i in range(data.shape[0]):
			test_ID.append(int(892+i))
	if args.lossf == 'mse':
		y_pred = np.where(model(data).array > 0.5, 1, 0)
	elif args.lossf == 'mse_sig':
		y_pred = np.where(F.sigmoid(model(data)).array > 0.5 ,1, 0)
	if args.model == 'ie':
		np.savetxt('./result/test/modeltest_{}_{}_{}_add{}_mdlno{}.csv'.format(args.day,args.model,args.data_model, args.add, no),np.vstack((np.array(test_ID,dtype = 'int32'),y_pred.reshape(-1,))).T, header='PassengerId,Survived',fmt='%d',delimiter=',',comments='')
	else:
		np.savetxt('./result/test/modeltest_{}_{}_{}_units{}_sum.csv'.format(args.day,args.model,args.data_model, args.mlp_units),np.vstack((np.array(test_ID,dtype = 'int32'),y_pred.reshape(-1,))).T, header='PassengerId,Survived',fmt='%d',delimiter=',',comments='')

def craft_titanic_sum(model_list, args, data):
	test_ID = []
	y_pred = 0
	for i in range(data.shape[0]):
			test_ID.append(int(892+i))

	if args.lossf == 'mse':
		for mod in model_list:
			y_pred += mod(data).array
	elif args.lossf == 'mse_sig':
		for mod in model_list:
			y_pred += F.sigmoid(mod(data)).array
	y_pred = y_pred/len(model_list)
	y_pred = np.where(y_pred > 0.5, 1, 0)
	if args.model == 'ie':
		np.savetxt('./result/test/modeltest_{}_{}_{}_add{}_sum.csv'.format(args.day,args.model,args.data_model,args.add),np.vstack((np.array(test_ID,dtype = 'int32'),y_pred.reshape(-1,))).T, header='PassengerId,Survived',fmt='%d',delimiter=',',comments='')
	else:
		np.savetxt('./result/test/modeltest_{}_{}_{}_units{}_sum.csv'.format(args.day,args.model,args.data_model, args.mlp_units),np.vstack((np.array(test_ID,dtype = 'int32'),y_pred.reshape(-1,))).T, header='PassengerId,Survived',fmt='%d',delimiter=',',comments='')

def under_sampling(y_data,x_data):
    # npの一次元データをもらって返す。Xは二次元
    negative = []
    positive = []
    for i in range(y_data.size):
        if y_data[i] == 0:
            negative.append(i)
        else:
            positive.append(i)
    
    if len(negative) > len(positive):
        y_under_no = random.sample(negative, k=len(positive))
        return y_data[y_under_no+positive],x_data[y_under_no+positive]
    elif len(positive) > len(negative):
        y_under_no = random.sample(positive, k=len(negative)) 
        return y_data[y_under_no+negative],x_data[y_under_no+negative]

def accuracy(y_true,y_score):
	score = accuracy_score(y_true, np.where(y_score.reshape(-1,) > 0.5, 1, 0))
	precision = precision_score(y_true, np.where(y_score.reshape(-1,) > 0.5, 1, 0))
	recall = recall_score(y_true, np.where(y_score.reshape(-1,) > 0.5, 1, 0))
	f1 = f1_score(y_true, np.where(y_score.reshape(-1,) > 0.5, 1, 0))
	AUC = roc_auc_score(y_true, y_score)
	print('精度:{:.3f}'.format(score))
	print('適合率:{:.3f}'.format(precision))
	print('再現率:{:.3f}'.format(recall))
	print('f-1値:{:.3f}'.format(f1))
	print("AUC_score:{}".format(AUC))
	# fpr, tpr, thresholds = roc_curve(y_true, y_score)
	# plt.plot(fpr, tpr, marker='o')
	# plt.xlabel('FPR: False positive rate')
	# plt.ylabel('TPR: True positive rate')
	# plt.grid()
	# plt.show()
	return score, precision, recall, f1, AUC

def print_r2(y_true, y_score, lossf):
	if lossf == 'mse_sig':
		y_score = F.sigmoid(y_score).array
	print("平均決定係数:{}".format(r2_score(y_true, y_score)))
	pass

def calc_r2(y_true, y_score, lossf):
	if lossf == 'mse_sig':
		y_score = F.sigmoid(y_score).array
	return r2_score(y_true, y_score)

def calc_ak_mse(y_true, y_score, lossf, valuesize):
	if lossf == 'mse_sig':
		y_score = F.sigmoid(y_score).array
	sigma_2 = mean_squared_error(y_true, y_score)
	sigma = np.sqrt(sigma_2)
	M = y_true.size
	return 2*M*math.log(np.sqrt(2*math.pi)*sigma)+M+2*(valuesize-1)

def display_null_importance(list, number, args):
	# ある特徴量に対する重要度を取得
	actual_imp = list[1][number]
	null_imp = [r[number] for r in list]
	del null_imp[:2]
	# 可視化
	fig, ax = plt.subplots(1, 1, figsize=(6, 4))
	a = ax.hist(null_imp, label="Null importances")
	ax.vlines(x=actual_imp, ymin=0, ymax=np.max(a[0]), color='r', linewidth=10, label='Real Target')
	ax.legend(loc="upper right")
	ax.set_title("Importance of {}".format(list[0][number]), fontweight='bold')
	plt.xlabel("Importance_{}".format(list[0][number]))
	fig.savefig("./result/picture/Null_imp_image/Null_add{}_lepoch{}_{}.png".format(args.add, args.loss_epoch,list[0][number]))

def cross_valid_custum(dataset, k, boot=1):
	kf = KFold(n_splits = k, shuffle = True)
	kf_list = []
	valuesize = dataset.shape[1]
	for train_index, test_index in kf.split(dataset):
		Y_test = dataset[test_index, 0]
		X1_test = dataset[test_index, 1:valuesize]
		X_test = chainer.datasets.TupleDataset(X1_test, Y_test)
		if boot == 1:
			Y_train = dataset[train_index, 0]
			X1_train = dataset[train_index, 1:valuesize]
			X_train = chainer.datasets.TupleDataset(X1_train, Y_train)
		else:
			max_object = np.amax(dataset[:,0])
			data_list = []
			for i in range(int(max_object)+1):
				data = dataset[train_index][dataset[train_index][:,0]==i]
				data_list.append(data[np.random.choice(np.array(range(len(data))), size = boot , replace=True)])
			data = np.random.permutation(np.vstack(tuple(data_list)))
			Y_train = data[0:data.shape[0], 0]
			X1_train = data[0:data.shape[0], 1:data.shape[1]]
			X_train = chainer.datasets.TupleDataset(X1_train, Y_train)
		kf_list.append((X_train, X_test))
	return kf_list

