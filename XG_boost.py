#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xgboost as xgb
import argparse
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_predict
from matplotlib import pyplot as plt
from sklearn import tree
import pickle
import sys
import copy
from sklearn.externals.six import StringIO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve,roc_auc_score, precision_recall_fscore_support, r2_score
from sklearn.model_selection import KFold
import shap
import calc
import mlp
import argparse
import pandas as pd
import numpy as np
import csv
import time
import pylab
from imblearn.under_sampling import ClusterCentroids

import chainer
from chainer import iterators
from chainer import optimizers
from chainer import serializers
from chainer.datasets import split_dataset_random
from chainer.optimizer_hooks import WeightDecay
"""XGBoost で特徴量の重要度を可視化するサンプルコード"""


def main():

        parser = argparse.ArgumentParser()
        parser.add_argument('--data_model', type=str, default='diabesets_v6', help='_3_3_pool_1_mnist_2class：CSVの入力ファイル指名')

        #初期値手動で変えるとき使用
        parser.add_argument('--initi',type=str, default='off', help='on or off')

        # k分割交差検証するときはk>1の整数。しないときはk=1にしてください。お願いします。あとk=1にしたらtrain_rateは0以上にしないとエラーが起きる
        parser.add_argument('--train_rate',type=float ,default=0.75, help='0~1の間trainの割合を決める.1ならテストなし')
        parser.add_argument('--k', type=int, default=4, help='k分割交差検証のkの値。1だと分割せずそれ以上の整数値だと分割する')
        parser.add_argument('--sampling', type=str, default='nashi', help='サンプリングの手法を選択')
        parser.add_argument('--k_test', type=int, default=0, help='k分割交差検証のtestデータをさらに作るなら１にして')
        parser.add_argument('--not_monotony', action='store_true',help='つけたら単調性なし')

        # データのディレクトリの場所のパラメータ
        parser.add_argument('--directri', type=str, default='sklearn_data', help='ディレクトリを指定')
        args = parser.parse_args()
        score_sum, precision_sum, recall_sum, f1_sum, AUC_sum = 0,0,0,0,0
        # read data
        print("===== read data =====", flush=True)
        print()

        #データの読み込み
        dataset = np.genfromtxt("./data/{}/{}.csv".format(args.directri, args.data_model), delimiter=",", filling_values=0).astype(np.float32)

        valuesize = dataset.shape[1]
        #Y = (dataset[1:dataset.shape[0], 0]).astype(np.int32)
        Y = (dataset[0:dataset.shape[0], 0])
        X1 = dataset[0:dataset.shape[0], 1:valuesize]

        #クラスタリングの選択
        if args.sampling == 'ClusterCentroids':
                X1,Y = ClusterCentroids(sampling_strategy=0.36 ,random_state=0).fit_resample(X1,Y) #sampling_strategy=low_sample/lot_sample
        elif args.sampling == 'SMOTE':
                X1,Y = SMOTE(random_state=0).fit_resample(X1, Y)
        elif args.sampling == 'RandomUnderSampler':
                X1,Y = RandomUnderSampler(random_state=0).fit_resample(X1, Y)
        else:
                print('サンプリングなし')
                pass

        diabeset_name = [np.array('Measure'),np.array(['age','sex','bmi','map','tc','ldl','hdl','tch','ltg','glu'])]


        # dataset = np.genfromtxt("./data/diabeset/diabeset_1_2_only.csv", delimiter=",", filling_values=0).astype(np.float32)
        # valuesize = dataset.shape[1]
        # Y = (dataset[1:dataset.shape[0], 0]).astype(np.int32)
        # X1 = dataset[1:dataset.shape[0], 1:valuesize]

        # X, y = diabeset_all._datasets[0], diabeset_all._datasets[1]
        # X_train, X_test, y_train, y_test = train_test_split(X1, Y,
        #                                                 test_size=0.2,
        #                                                 shuffle=True,
        #                                                 random_state=0,
        #                                                 )
        
        kf = KFold(n_splits= 4, shuffle = True)
        result_value = []
        shap_mean = []
        num = 0
        for i in range(1):
                defined_value = []
                defined_ak = []
                for train_index, test_index in kf.split(X1,Y):
                        num += 1
                        # 可視化のために特徴量の名前を渡しておく
                        dtrain = xgb.DMatrix(X1[train_index],
                                        label=Y[train_index],
                                        feature_names=diabeset_name[1].tolist())
                        dtest = xgb.DMatrix(X1[test_index], label=Y[test_index],
                                        feature_names=diabeset_name[1].tolist())
                        # dRtest = xgb.DMatrix(Rtest._datasets[0],
                        #                 feature_names=v_name[1].tolist())

                        # dall = xgb.DMatrix(X1[train_index],
                        #                 label=Y,
                        #                 feature_names=diabeset_name[1].tolist())

                        xgb_params = {
                                'max_depth':1,
                                'objective': 'reg:linear',
                                'eval_metric': 'rmse',
                                'min_child_weight':2,
                                'max_delta_step':6,
                                'alpha':1/6,

                        }

                        evals = [(dtrain, 'train'), (dtest, 'eval')]
                        evals_result = {}
                        bst = xgb.train(xgb_params,
                                dtrain,
                                num_boost_round=100,
                                early_stopping_rounds=10,
                                evals=evals,
                                evals_result=evals_result,
                                )

                        
                        # y_pred_proba = bst.predict(dtest)
                        # y_pred = np.where(y_pred_proba > 0.5, 1, 0)
                        # #acc = accuracy_score(y_test, y_pred)
                        # #print('Accuracy:', acc)

                        # test_ID = []
                        # for i in range(Rtest._length):
                        #         test_ID.append(int(892+i))

                        # y_pred = np.where(bst.predict(dRtest) > 0.5, 1, 0)

                        # np.savetxt('./result/test/modeltest_{}_{}_{}.csv'.format('12_4','XGBoost','allTitanic'),np.vstack((np.array(test_ID,dtype = 'int32'),y_pred)).T, header='PassengerId,Survived',fmt='%d',delimiter=',',comments='')

                
                        defined_value.append(calc.calc_r2(Y[test_index].tolist(), bst.predict(dtest).tolist()))
                        defined_value.append
                        # 学習の課程を折れ線グラフとしてプロットする
                        # train_metric = evals_result['train']['rmse']
                        # plt.plot(train_metric, label='train rmse')
                        # eval_metric = evals_result['eval']['rmse']
                        # plt.plot(eval_metric, label='eval rmse')
                        # plt.grid()
                        # plt.legend()
                        # plt.xlabel('rounds')
                        # plt.ylabel('rmse')
                        # plt.show()

                        # print('精度:{:.3f}'.format(accuracy_score(y, bst.predict(dall))))
                        # print('適合率:{:.3f}'.format(precision_score(y, bst.predict(dall))))
                        # print('再現率:{:.3f}'.format(recall_score(y, bst.predict(dall))))
                        # print('f-1値:{:.3f}'.format(f1_score(y, bst.predict(dall))))
                        # print("AUC_score:{}".format(roc_auc_score(Y[test_index], bst.predict(dtest))))
                        # fpr, tpr, thresholds = roc_curve(Y[test_index], bst.predict(dtest))
                        # plt.plot(fpr, tpr, marker='o')
                        # plt.xlabel('FPR: False positive rate')
                        # plt.ylabel('TPR: True positive rate')
                        # plt.grid()
                        # plt.show()

                        # score,precision,recall,f1,AUC = calc.accuracy(Y[test_index], bst.predict(dtest))

                        #k分割の精度の合計を作る
                        # score_sum += score
                        # precision_sum += precision
                        # recall_sum += recall
                        # f1_sum += f1
                        # AUC_sum += AUC
                        #print("決定係数:{}".format(r2_score(y, bst.predict(dall))))

                        # 性能向上に寄与する度合いで重要度をプロットする
                        _, ax = plt.subplots(figsize=(12, 4))
                        xgb.plot_importance(bst,
                                        ax=ax,
                                        importance_type='gain',
                                        show_values=True
                                        )
                        _.savefig("./result/picture/gain/gain_{}.png".format(num))
                        # plt.show()
                        shap.initjs()
                        explainer = shap.TreeExplainer(model=bst)
                        shap_values = explainer.shap_values(X=dtest)
                        shap_mean.append(np.mean(np.array(abs(shap_values)),axis=0))
                        # shap.summary_plot(shap_values, X1[test_index], plot_type="bar")
                print("平均shap値：{}".format(np.mean(np.array(shap_mean),axis=0)))
                print("平均決定係数:{}".format(sum(defined_value)/4))
                result_value.append(sum(defined_value)/4)
                # print('平均精度')
                # print('精度:{:.3f}'.format(score_sum/5))
                # print('適合率:{:.3f}'.format(precision_sum/5))
                # print('再現率:{:.3f}'.format(recall_sum/5))
                # print('f-1値:{:.3f}'.format(f1_sum/5))
                # print("AUC_score:{}".format(AUC_sum/5))
        print(result_value)



if __name__ == '__main__':
        main()