import cv2
import gzip
import argparse
import numpy as np
import os
import random
import pickle
import copy
import chainer
from chainer.datasets import tuple_dataset
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def make2class(data, class0, class1, noiseFlag):
    image = [] # x
    label = []
    dataset = copy.deepcopy(data) # 参照渡し -> train_allの上書き防止
    for dt in dataset:
        img = dt[0]
        lbl = dt[1]
        if lbl in class0 or lbl in class1:
            if noiseFlag:
                for i in range(28):
                    for j in range(28):
                        r = random.randint(0, 2)
                        if r == 0:
                            img[0][i][j] = 1.0
                        elif r == 1:
                            img[0][i][j] = 0.0
            if lbl in class0:
                label.append(0)
                image.append(img)
            elif lbl in class1:
                label.append(1)
                image.append(img)
    return chainer.datasets.TupleDataset(image, label)

def make1class(data, class0, noiseFlag):
    image = [] # x
    label = []
    dataset = copy.deepcopy(data) # 参照渡し -> train_allの上書き防止
    for dt in dataset:
        img = dt[0]
        lbl = dt[1]
        if lbl == class0:
            if noiseFlag:
                for i in range(28):
                    for j in range(28):
                        r = random.randint(0, 2)
                        if r == 0:
                            img[0][i][j] = 1.0
                        elif r == 1:
                            img[0][i][j] = 0.0
            if lbl == class0:
                label.append(class0)
                image.append(img)
    return chainer.datasets.TupleDataset(image, label)

def make1class_3_3(data, class0, noiseFlag):
    image = [] # x
    label = []
    dataset = copy.deepcopy(data) # 参照渡し -> train_allの上書き防止
    for i in range(3):
		if i == 2:
			x = F.average_pooling_2d(x, 2, stride =2)
		else:
			x = F.average_pooling_2d(x, 2)
    for dt in dataset:
        img = dt[0]
        lbl = dt[1]
        if lbl == class0:
            if noiseFlag:
                for i in range(28):
                    for j in range(28):
                        r = random.randint(0, 2)
                        if r == 0:
                            img[0][i][j] = 1.0
                        elif r == 1:
                            img[0][i][j] = 0.0
            if lbl == class0:
                label.append(class0)
                image.append(img)
    return chainer.datasets.TupleDataset(image, label)




def shuffle(data1, data2):
    shuffle_data1 = data1
    shuffle_data2 = data2
    random.shuffle(shuffle_data1)
    random.shuffle(shuffle_data2)

    train = data1
    test = data2
    return train, test

def main():

    with open('./data/mnist/mnist.pkl', mode="rb") as f:
        train_all, test_all = pickle.load(f)
    
    
    # # debug
    # train_all = train_all[1:1000]
    # test_all = test_all[1:1000] 

    noiseFlag = False


    train = []
    
    print("- train1" , flush = True, end="")
    train.append(make1class(train_all, 0, noiseFlag))  # task1's train data
    print("_num: {}".format(len(train[0])))

    print("- train1" , flush = True, end="")
    train.append(make1class(train_all, 1, noiseFlag))  # task1's train data
    print("_num: {}".format(len(train[1])))

    print("- train1" , flush = True, end="")
    train.append(make1class(train_all, 2, noiseFlag))  # task1's train data
    print("_num: {}".format(len(train[2])))
    
    print("- train1" , flush = True, end="")
    train.append(make1class(train_all, 3, noiseFlag))  # task1's train data
    print("_num: {}".format(len(train[3])))

    print("- train1" , flush = True, end="")
    train.append(make1class(train_all, 4, noiseFlag))  # task1's train data
    print("_num: {}".format(len(train[4])))

    print("- train1" , flush = True, end="")
    train.append(make1class(train_all, 5, noiseFlag))  # task1's train data
    print("_num: {}".format(len(train[5])))

    print("- train1" , flush = True, end="")
    train.append(make1class(train_all, 6, noiseFlag))  # task1's train data
    print("_num: {}".format(len(train[6])))

    print("- train1" , flush = True, end="")
    train.append(make1class(train_all, 7, noiseFlag))  # task1's train data
    print("_num: {}".format(len(train[7])))

    print("- train1" , flush = True, end="")
    train.append(make1class(train_all, 8, noiseFlag))  # task1's train data
    print("_num: {}".format(len(train[8])))

    print("- train1" , flush = True, end="")
    train.append(make1class(train_all, 9, noiseFlag))  # task1's train data
    print("_num: {}".format(len(train[9])))

    test = []
    
    print("- test1" , flush = True, end="")    
    test.append(make1class(test_all, 0, noiseFlag))  # task1's test data
    print("_num: {}".format(len(test[0])))

    print("- test1" , flush = True, end="")
    test.append(make1class(test_all, 1, noiseFlag))  # task1's test data
    print("_num: {}".format(len(test[1])))

    print("- test1" , flush = True, end="")
    test.append(make1class(test_all, 2, noiseFlag))  # task1's test data
    print("_num: {}".format(len(test[2])))
    
    print("- test1" , flush = True, end="")
    test.append(make1class(test_all, 3, noiseFlag))  # task1's test data
    print("_num: {}".format(len(test[3])))

    print("- test1" , flush = True, end="")
    test.append(make1class(test_all, 4, noiseFlag))  # task1's test data
    print("_num: {}".format(len(test[4])))

    print("- test1" , flush = True, end="")
    test.append(make1class(test_all, 5, noiseFlag))  # task1's test data
    print("_num: {}".format(len(test[5])))

    print("- test1" , flush = True, end="")
    test.append(make1class(test_all, 6, noiseFlag))  # task1's test data
    print("_num: {}".format(len(test[6])))

    print("- test1" , flush = True, end="")
    test.append(make1class(test_all, 7, noiseFlag))  # task1's test data
    print("_num: {}".format(len(test[7])))

    print("- test1" , flush = True, end="")
    test.append(make1class(test_all, 8, noiseFlag))  # task1's test data
    print("_num: {}".format(len(test[8])))

    print("- test1" , flush = True, end="")
    test.append(make1class(test_all, 9, noiseFlag))  # task1's test data
    print("_num: {}".format(len(test[9])))


    train_num = []
    
    for num in range(10):
        train_ = []
        for i in range(len(train[num])):
            train_.append(train[num][i][0].reshape(-1))
        train_num.append(np.array(train_))
    

    test_num = []
    
    for num in range(10):
        test_ = []
        for i in range(len(test[num])):
            test_.append(test[num][i][0].reshape(-1))
        test_num.append(np.array(test_))


    pkl_train_data = []
    pkl_test_data = []

    pkl_train_data.append(train_num)
    pkl_test_data.append(test_num)
    pickle.dump(pkl_train_data, open("pcd_train_mnist_1class.pkl", "wb"), -1)
    pickle.dump(pkl_test_data, open("pcd_test_mnist_1class.pkl", "wb"), -1)

    
if __name__ == '__main__':
	main()