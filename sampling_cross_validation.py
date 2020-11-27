import numpy as np
import chainer
from chainer import iterators, optimizers, initializers
import chainer.functions as F
import chainer.links as L
from chainer.datasets import split_dataset_random,get_cross_validation_datasets
from chainer.dataset import concat_examples
from matplotlib import pyplot
import pickle
from imblearn.under_sampling import RandomUnderSampler
import random
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
# 正例:負例 = 1:9でサンプル生成
X, y = make_classification(n_samples=100000,
                           n_features=5,
                           n_classes=2,
                           weights=[0.9, 0.1],
                           random_state=42)
# 学習・テストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0)

from imblearn.under_sampling import RandomUnderSampler
sampler = RandomUnderSampler(sampling_strategy={0: y_train.sum(), 1: y_train.sum()}, random_state=42)


dataset = np.genfromtxt("./data/car/car_1_2_only.csv", delimiter=",", filling_values=0).astype(np.float32)
valuesize = dataset.shape[1]
Y = (dataset[1:dataset.shape[0], 0]).astype(np.int32)
X1 = dataset[1:dataset.shape[0], 1:valuesize]

X_smoteenn, y_smoteenn=SMOTE(random_state=16).fit_resample(X1, Y)


X = chainer.datasets.TupleDataset(X1, Y)
cross = get_cross_validation_datasets(X,10)



# def under_sampling(y_data,x_data):
#     # npの一次元データをもらって返す。Xは二次元
#     negative = []
#     positive = []
#     for i in range(y_data.size):
#         if y_data[i] == 0:
#             negative.append(i)
#         else:
#             positive.append(i)
    
#     if len(negative) > len(positive):
#         y_under_no = random.sample(negative, k=len(positive))
#         return y_data[y_under_no+positive],x_data[y_under_no+positive]
#     elif len(positive) > len(negative):
#         y_under_no = random.sample(positive, k=len(negative)) 
#         return y_data[y_under_no+negative],x_data[y_under_no+negative]

# def data_sampler(args):
#     k = args.k

#     dataset = np.genfromtxt("./data/{}/{}.csv".format(args.directri, args.data_model), delimiter=",", filling_values=0).astype(np.float32)
#     valuesize = dataset.shape[1]
#     Y = (dataset[1:cardata.shape[0], 0]).astype(np.int32)
#     X1 = dataset[1:cardata.shape[0], 1:valuesize]

#     if args.sampling == 'under_sampling':
#         Y,X1 = under_sampling(Y, X1)
#     else:
#         pass
    
#     datasize = Y.shape[0]
#     Randomlist = np.random.permutation(np.array(range(datasize)))
#     validation_no = datasize*args.validation_rate
