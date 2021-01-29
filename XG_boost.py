#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from matplotlib import pyplot as plt
from sklearn import tree
import pickle
import sys
import copy
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals.six import StringIO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve,roc_auc_score, precision_recall_fscore_support, r2_score, mean_squared_error, mean_absolute_error
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

import random as rnd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import chainer
from chainer import iterators
from chainer import optimizers
from chainer import serializers
from chainer.datasets import split_dataset_random
from chainer.optimizer_hooks import WeightDecay
"""XGBoost で特徴量の重要度を可視化するサンプルコード"""


def main():

        parser = argparse.ArgumentParser()
        parser.add_argument('--data_model', type=str, default='Titanic_train_3pop_df', help='_3_3_pool_1_mnist_2class：CSVの入力ファイル指名')

        #初期値手動で変えるとき使用
        parser.add_argument('--initi',type=str, default='off', help='on or off')

        # k分割交差検証するときはk>1の整数。しないときはk=1にしてください。お願いします。あとk=1にしたらtrain_rateは0以上にしないとエラーが起きる
        parser.add_argument('--k', type=int, default=5, help='k分割交差検証のkの値。1だと分割せずそれ以上の整数値だと分割する')
        parser.add_argument('--sampling', type=str, default='nashi', help='サンプリングの手法を選択')
        parser.add_argument('--k_test', type=int, default=0, help='k分割交差検証のtestデータをさらに作るなら１にして')
        parser.add_argument('--not_monotony', action='store_true',help='つけたら単調性なし')
        parser.add_argument('--model', type=str, default='XGboost',help='model selection')

        # データのディレクトリの場所のパラメータ
        parser.add_argument('--directri', type=str, default='Titanic', help='ディレクトリを指定')
        args = parser.parse_args()
        score_sum, precision_sum, recall_sum, f1_sum, AUC_sum = 0,0,0,0,0
        # read data
        print("===== read data =====", flush=True)
        print()

        #データの読み込み
        data = pd.read_csv("./data/{}/{}.csv".format(args.directri, args.data_model))
        #data = np.genfromtxt("./data/{}/{}.csv".format(args.directri, args.data_model), delimiter=",", filling_values=0).astype(np.float32)

        # valuesize = dataset.shape[1]
        # #Y = (dataset[1:dataset.shape[0], 0]).astype(np.int32)
        # Y = (dataset[0:dataset.shape[0], 0])
        # X1 = dataset[0:dataset.shape[0], 1:valuesize]

        # #クラスタリングの選択
        # if args.sampling == 'ClusterCentroids':
        #         X1,Y = ClusterCentroids(sampling_strategy=0.36 ,random_state=0).fit_resample(X1,Y) #sampling_strategy=low_sample/lot_sample
        # elif args.sampling == 'SMOTE':
        #         X1,Y = SMOTE(random_state=0).fit_resample(X1, Y)
        # elif args.sampling == 'RandomUnderSampler':
        #         X1,Y = RandomUnderSampler(random_state=0).fit_resample(X1, Y)
        # else:
        #         print('サンプリングなし')
        #         pass

        kf = KFold(n_splits= args.k, shuffle = True)
        result_value = []
        mod_train_mse = []
        mod_train_mae = []
        mod_train_R2 = []
        mod_test_mse = []
        mod_test_mae = []
        mod_test_R2 = []
        mod_gain = []
        mod_shap = []
        defined_value = []
        defined_ak = []
        num = 0
        if args.model == "XGboost":
                import xgboost as xgb
                for train_index, test_index in kf.split(data):
                        shap_mean = []
                        gain_mean = []
                        num += 1
                        train_df = data.iloc[train_index]
                        test_df  = data.iloc[test_index]
                        X_train = train_df.drop("Y", axis=1)
                        y_train = train_df["Y"]
                        X_test  = test_df.drop("Y", axis=1)
                        y_test  = test_df["Y"]
                        # 可視化のために特徴量の名前を渡しておく
                        dtrain = xgb.DMatrix(X_train,
                                        label=y_train,
                                        feature_names=X_train.columns.base.tolist())
                        dtest = xgb.DMatrix(X_test, label=y_test,
                                        feature_names=X_train.columns.base.tolist())
                        # dRtest = xgb.DMatrix(Rtest._datasets[0],
                        #                 feature_names=v_name[1].tolist())

                        # dall = xgb.DMatrix(X1[train_index],
                        #                 label=Y,
                        #                 feature_names=diabeset_name[1].tolist())

                        xgb_params = {
                                'max_depth':20,
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

                        # defined_value.append(calc.calc_r2(Y[test_index].tolist(), bst.predict(dtest).tolist()))
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
                        gain_score = bst.get_score(importance_type='gain')
                        for j in X_train.columns:
                                print(j)
                                try:
                                        gain_mean.append(gain_score[j])
                                except KeyError:
                                        gain_mean.append(0)
                        # plt.show()
                        shap.initjs()
                        explainer = shap.TreeExplainer(model=bst)
                        shap_values = explainer.shap_values(X=dtest)
                        shap_mean.append(np.mean(np.array(abs(shap_values)),axis=0).tolist())
                        #shap.summary_plot(shap_values, X_test, plot_type="bar")
                        mod_gain.append(gain_mean)
                        mod_shap.append(shap_mean[0])

                        train_pred = bst.predict(dtrain)
                        test_pred = bst.predict(dtest)

                        train_mse    = mean_squared_error(y_train, train_pred)

                        train_mae    = mean_absolute_error(y_train, train_pred)

                        train_scores = r2_score(y_train, train_pred)

                        test_mse    = mean_squared_error(y_test, test_pred)

                        test_mae    = mean_absolute_error(y_test, test_pred)

                        test_scores = r2_score(y_test, test_pred)

                        print("train:")
                        print("mse train:{}".format(train_mse))
                        print("mae test :{}".format(train_mae))
                        print("R2:{}".format(train_scores))
                        print("test:")
                        print("mse linear:{}".format(test_mse))
                        print("mae linear:{}".format(test_mae))
                        print("R2:{}".format(test_scores))
                        mod_train_mse.append(train_mse)
                        mod_train_mae.append(train_mae)
                        mod_train_R2.append(train_scores)
                        mod_test_mse.append(test_mse)
                        mod_test_mae.append(test_mae)
                        mod_test_R2.append(test_scores)
                        
                mod_gain = np.array(mod_gain)
                mod_shap = np.array(mod_shap)
                print("gain")
                for l in np.mean(mod_gain, axis=0):
                        print(l)
                #print(np.mean(mod_gain, axis=0))
                print("shap")
                for l in np.mean(mod_shap, axis=0):
                        print(l)
                #print(np.mean(mod_shap, axis=0))
                print("train")
                print("平均絶対誤差の平均：{}".format(sum(mod_train_mae)/args.k))
                print("平均二乗誤差の平均：{}".format(sum(mod_train_mse)/args.k))
                print("平均決定係数:{}".format(sum(mod_train_R2)/args.k))
                print("test")
                print("平均絶対誤差の平均：{}".format(sum(mod_test_mae)/args.k))
                print("平均二乗誤差の平均：{}".format(sum(mod_test_mse)/args.k))
                print("平均決定係数:{}".format(sum(mod_test_R2)/args.k))
                # print("平均二乗誤差の平均：{}".format(sum(mod_mse)/args.k))
                # print("平均絶対誤差の平均：{}".format(sum(mod_mae)/args.k))
                # print("平均shap値：{}".format(np.mean(np.array(shap_mean),axis=0)))
                # print("平均決定係数:{}".format(sum(defined_value)/args.k))
                # result_value.append(sum(defined_value)/args.k)
                # # print('平均精度')
                # # print('精度:{:.3f}'.format(score_sum/5))
                # # print('適合率:{:.3f}'.format(precision_sum/5))
                # # print('再現率:{:.3f}'.format(recall_sum/5))
                # # print('f-1値:{:.3f}'.format(f1_sum/5))
                # # print("AUC_score:{}".format(AUC_sum/5))
                # print(result_value)
        elif args.model == "svm":
                from sklearn.svm import SVR

                for train_index, test_index in kf.split(data):
                        num += 1
                        train_df = data.iloc[train_index]
                        test_df  = data.iloc[test_index]
                        X_train = train_df.drop("Y", axis=1)
                        y_train = train_df["Y"]
                        X_test  = test_df.drop("Y", axis=1)
                        y_test  = test_df["Y"]
                        reg_linear = SVR(kernel='linear', C=1, epsilon=0.1)
                        reg_poly = SVR(kernel='poly', C=1, epsilon=0.1)
                        reg_rbf = SVR(kernel='rbf', C=1, epsilon=0.1)

                        reg_linear.fit(X_train, np.ravel(y_train))
                        reg_poly.fit(X_train, np.ravel(y_train))
                        reg_rbf.fit(X_train, np.ravel(y_train))

                        pre_train_linear = reg_linear.predict(X_train)
                        pre_train_poly = reg_poly.predict(X_train)
                        pre_train_rbf = reg_rbf.predict(X_train)

                        pre_test_linear = reg_linear.predict(X_test)
                        pre_test_poly = reg_poly.predict(X_test)
                        pre_test_rbf = reg_rbf.predict(X_test)

                        train_mse    = [mean_squared_error(y_train, pre_train_linear),
                        mean_squared_error(y_train, pre_train_poly),
                        mean_squared_error(y_train, pre_train_rbf)]

                        train_mae    = [mean_absolute_error(y_train, pre_train_linear),
                        mean_absolute_error(y_train, pre_train_poly),
                        mean_absolute_error(y_train, pre_train_rbf)]

                        train_scores = [reg_linear.score(X_train, y_train),
                        reg_poly.score(X_train, y_train),
                        reg_rbf.score(X_train, y_train)]

                        test_mse    = [mean_squared_error(y_test, pre_test_linear),
                        mean_squared_error(y_test, pre_test_poly),
                        mean_squared_error(y_test, pre_test_rbf)]

                        test_mae    = [mean_absolute_error(y_test, pre_test_linear),
                        mean_absolute_error(y_test, pre_test_poly),
                        mean_absolute_error(y_test, pre_test_rbf)]

                        test_scores = [reg_linear.score(X_test, y_test),
                        reg_poly.score(X_test, y_test),
                        reg_rbf.score(X_test, y_test)]
                        # plt.bar(("Linear", "poly", "RBF"), scores)
                        # plt.xlabel("Kernel")
                        # plt.ylabel("$R^2$ score")
                        # plt.show()

                        print("train:linear:poly:rbf:")
                        print("mse train:{}".format(train_mse))
                        print("mae test :{}".format(train_mae))
                        print("R2:{}".format(train_scores))
                        print("test:linear:poly:rbf:")
                        print("mse linear:{}".format(test_mse))
                        print("mae linear:{}".format(test_mae))
                        print("R2:{}".format(test_scores))
                        mod_train_mse.append(train_mse)
                        mod_train_mae.append(train_mae)
                        mod_train_R2.append(train_scores)
                        mod_test_mse.append(test_mse)
                        mod_test_mae.append(test_mae)
                        mod_test_R2.append(test_scores)
                print("train")
                print("平均二乗誤差の平均：{}".format(np.sum(np.array(mod_train_mse), axis=0)/args.k))
                print("平均絶対誤差の平均：{}".format(np.sum(np.array(mod_train_mae), axis=0)/args.k))
                print("平均決定係数:{}".format(np.sum(np.array(mod_train_R2), axis=0)/args.k))
                print("test")
                print("平均二乗誤差の平均：{}".format(np.sum(np.array(mod_test_mse), axis=0)/args.k))
                print("平均絶対誤差の平均：{}".format(np.sum(np.array(mod_test_mae), axis=0)/args.k))
                print("平均決定係数:{}".format(np.sum(np.array(mod_test_R2), axis=0)/args.k))
        elif args.model == "dtree":
                from sklearn.tree import DecisionTreeRegressor
                for train_index, test_index in kf.split(data):
                        train_df = data.iloc[train_index]
                        test_df  = data.iloc[test_index]
                        X_train = train_df.drop("Y", axis=1)
                        y_train = train_df["Y"]
                        X_test  = test_df.drop("Y", axis=1)
                        y_test  = test_df["Y"]
                        decision_tree = DecisionTreeRegressor(max_depth=2)
                        decision_tree.fit(X_train, y_train)
                        train_pred = decision_tree.predict(X_train)
                        test_pred  = decision_tree.predict(X_test)

                        train_mse    = mean_squared_error(y_train, train_pred)

                        train_mae    = mean_absolute_error(y_train, train_pred)

                        train_scores = decision_tree.score(X_train,y_train)

                        test_mse    = mean_squared_error(y_test, test_pred)

                        test_mae    = mean_absolute_error(y_test, test_pred)

                        test_scores = decision_tree.score(X_test,y_test)

                        print("train:")
                        print("mse train:{}".format(train_mse))
                        print("mae test :{}".format(train_mae))
                        print("R2:{}".format(train_scores))
                        print("test:")
                        print("mse linear:{}".format(test_mse))
                        print("mae linear:{}".format(test_mae))
                        print("R2:{}".format(test_scores))
                        mod_train_mse.append(train_mse)
                        mod_train_mae.append(train_mae)
                        mod_train_R2.append(train_scores)
                        mod_test_mse.append(test_mse)
                        mod_test_mae.append(test_mae)
                        mod_test_R2.append(test_scores)
                print("train")
                print("平均二乗誤差の平均：{}".format(sum(mod_train_mse)/args.k))
                print("平均絶対誤差の平均：{}".format(sum(mod_train_mae)/args.k))
                print("平均決定係数:{}".format(sum(mod_train_R2)/args.k))
                print("test")
                print("平均二乗誤差の平均：{}".format(sum(mod_test_mse)/args.k))
                print("平均絶対誤差の平均：{}".format(sum(mod_test_mae)/args.k))
                print("平均決定係数:{}".format(sum(mod_test_R2)/args.k))
        elif args.model == "rforest":
                from sklearn.tree import DecisionTreeRegressor
                for train_index, test_index in kf.split(data):
                        train_df = data.iloc[train_index]
                        test_df  = data.iloc[test_index]
                        X_train = train_df.drop("Y", axis=1)
                        y_train = train_df["Y"]
                        X_test  = test_df.drop("Y", axis=1)
                        y_test  = test_df["Y"]
                        random_forest = RandomForestRegressor(n_estimators=100, max_depth=2)
                        random_forest.fit(X_train, y_train)
                        train_pred = random_forest.predict(X_train)
                        test_pred = random_forest.predict(X_test)
                        
                        train_mse    = mean_squared_error(y_train, train_pred)

                        train_mae    = mean_absolute_error(y_train, train_pred)

                        train_scores = random_forest.score(X_train, y_train)

                        test_mse    = mean_squared_error(y_test, test_pred)

                        test_mae    = mean_absolute_error(y_test, test_pred)

                        test_scores = random_forest.score(X_test, y_test)

                        print("train:")
                        print("mse train:{}".format(train_mse))
                        print("mae test :{}".format(train_mae))
                        print("R2:{}".format(train_scores))
                        print("test:")
                        print("mse linear:{}".format(test_mse))
                        print("mae linear:{}".format(test_mae))
                        print("R2:{}".format(test_scores))
                        mod_train_mse.append(train_mse)
                        mod_train_mae.append(train_mae)
                        mod_train_R2.append(train_scores)
                        mod_test_mse.append(test_mse)
                        mod_test_mae.append(test_mae)
                        mod_test_R2.append(test_scores)
                print("train")
                print("平均絶対誤差の平均：{}".format(sum(mod_train_mae)/args.k))
                print("平均二乗誤差の平均：{}".format(sum(mod_train_mse)/args.k))
                print("平均決定係数:{}".format(sum(mod_train_R2)/args.k))
                print("test")
                print("平均絶対誤差の平均：{}".format(sum(mod_test_mae)/args.k))
                print("平均二乗誤差の平均：{}".format(sum(mod_test_mse)/args.k))
                print("平均決定係数:{}".format(sum(mod_test_R2)/args.k))

        



if __name__ == '__main__':
        main()