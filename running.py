from __future__ import division

import numpy as np
# import cupy as np
import pandas as pd
import itertools
import math
import chainer
from chainer import initializers
import chainer.links as L
import chainer.functions as F
from chainer.dataset import concat_examples
from chainer.cuda import to_cpu
from chainer import training
from chainer import Variable
from chainer import optimizers
import copy
import time
# import relu_1
import shape_ver2
import calc

from itertools import chain,combinations

def run(self, train_iter, test_iter, optimizer, elapsed_time, start, args):
    # loos, acc. w 保存箱
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    out_pre_w = []
    out_post_w = []
    shape_box = []

    test_loss_point = 10000000000000000000
    loss_count = 0
    epoch_num = 0

    # 変数分の重み箱作成（記録用）
    for i in range(len(self.ie_data[0])*2):
        out_pre_w.append([])
    for i in range(len(self.hh)):
        out_post_w.append([])

    if self.args.out == 2:
        for i in range(len(self.hh)-1):
            out_post_w[i].append([])
            out_post_w[i].append([])

    print("===== read epoch =====", flush=True)

    if self.args.gpu_id >= 0:
        self.to_gpu(self.args.gpu_id)
    
    # epochごとの学習
    for epoch in range(self.args.epoch):
        err_temp = 0
        acc_temp = 0

        # バッチサイズごとの学習
        for i in range(0, len(train_iter.dataset), self.args.batch_size): # batch size loop                
            # for
            elapsed_time_bt = []
            start_bt = time.time()
            train_batch = train_iter.next()
            x, target = concat_examples(train_batch, self.args.gpu_id)
            if args.data == "car":
                target = F.reshape(target,(self.args.batch_size, -1))
            x = x.reshape(self.args.batch_size, len(self.ie_data[0]))
            y = self(x, self.args.tnorm)
            # print("batch: {} / {}".format(len(train_iter.dataset), i), end="")
            
            # 学習前の値を取得
            if i == 0 and epoch == 0:
                inti_model = copy.deepcopy(self)
                
                # 初期値の重みを取得
                # 入力層ー中間層の重みを取得（記録用）
                if self.args.fmodel == "random" or self.args.fmodel == "init" or self.args.pre_shoki == "units":
                    pass
                else:
                    for o in range(0, len(self.ie_data[0])*2):
                        if o % 2 == 0:
                            out_pre_w[o].append(inti_model.l[int(o/2)].W.data[0][0])
                        else:
                            out_pre_w[o].append(inti_model.l[int(o/2)].b.data[0])
                
                # 中間層ー出力層の重みを取得（記録用）
                if self.args.out == 2:
                    for p in range(len(self.hh)-1):            
                        out_post_w[p][0].append(inti_model.lt.W.data[0][p])
                        out_post_w[p][1].append(inti_model.lt.W.data[1][p])
                else:
                    for q in range(len(self.hh)-1):            
                        out_post_w[q].append(inti_model.lt.W.data[0][q])
                    out_post_w[len(self.hh)-1].append(inti_model.lt.b.data[0])

            # 損失関数の定義
            # 出力２のとき
            if self.args.out == 2:
                loss = F.softmax_cross_entropy(y, target)
                acc = F.accuracy(y, target)
                # softmaxで確率でだした後にｔのインデックスの方がloge(y)で出力
                # loss = F.softmax_cross_entropy(y, target)
            else:
                # 出力１の時
                if self.args.lossf == "ent":
                    loss = F.sigmoid_cross_entropy(y,F.reshape(target,(self.args.batch_size,1)))
                    y_ = F.concat((1-F.sigmoid(y), F.sigmoid(y)),axis=1)
                    acc = F.accuracy(y_, target.astype('int32'))
                elif self.args.lossf == "mse_sig":
                    loss  = F.mean_squared_error(y, F.reshape(target,(self.args.batch_size,1)).data.astype(np.float32))#出力にsigmoidつけた
                    y_ = F.concat((1-y, y),axis=1)
                    acc = F.accuracy(y_.array, target.astype('int32'))
                elif self.args.lossf == "mse":
                    loss  = F.mean_squared_error(y, F.reshape(target,(self.args.batch_size,1)).data.astype(np.float32))
                    y_ = [1 if r > 0.5 else 0 for r in y.data]
                    tf = np.array(y_) == np.array(list(target))
                    acc = Variable(np.array(np.count_nonzero(tf) / self.args.batch_size))
                # loss  = F.mean_squared_error(y, F.reshape(target,(self.args.batch_size,1)).data.astype(np.float32))
                # yのargmaxとtのインデックスが一致したら+1
                    

            #if self.args.epo == 0:#134
            
            self.cleargrads()#勾配のリセット
            loss.backward()#勾配の計算
            optimizer.update()#更新

            if self.args.norm == "lt" and epoch > 100:
                for i in range(self.lt.W.array.shape[1]):
                    if abs(self.lt.W.data[0,i]) < 0.01:
                        self.lt.W.data[0,i] = 0
            # _w = copy.deepcopy(self.lt.W.data)
            
            # 単調性をありにするとき
            if not self.args.not_monotony:
                calc.monotony(self, train_iter, args)

            err_temp += float(loss.data)
            if self.args.acc_info == "on":
                acc_temp += float(acc.data)

            elapsed_time_bt.append(time.time() - start_bt)
            # print(', time:{}'.format(elapsed_time_bt[0]))
            
        length = math.ceil(len(train_iter.dataset)/self.args.batch_size)
        train_loss.append(err_temp/length)
        if args.acc_info == 'on':
            train_acc.append(acc_temp/length)

        if self.args.acc_info == "on":
            print("epoch: {} train_loss: {:.4f} train_acc: {:.4f}".format(epoch, err_temp/length, acc_temp/length), end ="")
        else:
            print("epoch: {} train_loss: {:.4f}".format(epoch, err_temp/length), end ="")


        
        # テスト学習開始（テストデータで誤差を見る）
        test_batch = test_iter.next()
        x, target = concat_examples(test_batch, self.args.gpu_id)
        if args.data == "car":
            target = F.reshape(target,(len(train_batch), -1))
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = x.reshape(len(test_batch), len(self.ie_data[0]))
            # x = x.reshape(len(test_batch),len(self.ie_data[0]))
            y = self(x, self.args.tnorm)
        
        if args.data == "car":
            loss = F.mean_squared_error(y, target)
        elif args.data == "mnist":
            if self.args.out == 2:
                loss = F.softmax_cross_entropy(y, target)
                acc = F.accuracy(y, target)
                # softmaxで確率でだした後にｔのインデックスの方がloge(y)で出力
                # loss = F.softmax_cross_entropy(y, target)
            else:
                if self.args.lossf == "ent":
                    loss = F.sigmoid_cross_entropy(y,F.reshape(target,(len(test_batch),1)))
                    y_ = F.concat((1-F.sigmoid(y), F.sigmoid(y)),axis=1)
                    acc = F.accuracy(y_, target.astype('int32'))
                elif self.args.lossf == "mse_sig":
                    loss  = F.mean_squared_error(y, F.reshape(target,(len(test_batch),1)).data.astype(np.float32))
                    y_ = F.concat((1-y, y),axis=1)
                    acc = F.accuracy(y_, target.astype('int32'))
                elif self.args.lossf == "mse":
                    loss  = F.mean_squared_error(y, F.reshape(target,(len(test_batch),1)).data.astype(np.float32))
                    y_ = [1 if s > 0.5 else 0 for s in y.data]
                    tf = np.array(y_) == np.array(list(target))
                    acc = Variable(np.array(np.count_nonzero(tf) /len(test_batch)))
                # loss  = F.mean_squared_error(y, F.reshape(target,(self.args.batch_size,1)).data.astype(np.float32))
                # yのargmaxとtのインデックスが一致したら+1
                    
        # get shape
        ww_shape = []
        # ０いれる
        ww_shape.append(0)
        for ss in range(len(out_post_w)):
            ww_shape.append(out_post_w[ss][-1])

        # print(ww_shape)
        if args.matrixtype == 6:
            shape = shape_ver2.get_shape_2(ww_shape, self.hh)
        else:
            shape = shape_ver2.get_shape(ww_shape, len(self.ie_data[0]), args)
        shape_box.append(shape)
        # print()

        test_loss.append(float(to_cpu(loss.data)))
        if args.acc_info == 'on':
            test_acc.append(float(to_cpu(acc.data)))
       
        if self.args.acc_info == "on":
            print(" test_loss: {:.4f} test_acc: {:.4f}".format(float(to_cpu(loss.data)), float(to_cpu(acc.data))))
        else:
            print(" test_loss: {:.4f}".format(float(to_cpu(loss.data))))
        test_iter.reset()

        

        # epochごとに重み確保
        if self.args.fmodel == "random" or self.args.fmodel == "init" or self.args.pre_shoki == "units":
            pass
        else:
            for t in range(0, len(self.ie_data[0])*2):
                if t % 2 == 0:
                    out_pre_w[t].append(self.l[int(t/2)].W.data[0][0])
                else:
                    out_pre_w[t].append(self.l[int(t/2)].b.data[0])

        if self.args.out == 2:
            for u in range(len(self.hh)-1):            
                out_post_w[u][0].append(self.lt.W.data[0][u])
                out_post_w[u][1].append(self.lt.W.data[1][u])
        else:
            for w in range(len(self.hh)-1):
                out_post_w[w].append(self.lt.W.data[0][w])
            out_post_w[len(self.hh)-1].append(self.lt.b.data[0])

        #過学習する前に止める
        if test_loss_point > float(to_cpu(loss.data)):
            test_loss_point = float(to_cpu(loss.data))
            loss_count = 0
        elif loss_count == self.args.loss_epoch:
            break
        else:
            loss_count += 1
        # limit_train 
        max_epoch = 0
        last_train = float(to_cpu(acc.data)) #acc_temp/len(train_iter.dataset)

        epoch_num = epoch
        
        if last_train > args.limit:
            max_epoch = epoch
            break

    elapsed_time.append(time.time() - start)
    print("===== read time =====", flush=True)
    print('time:{}'.format(elapsed_time[0]))
    # train_loss, test_loss, w_b = model.train_model(train_iter, test_iter, optimizer)
    
    # 学習結果をまとめる（記録用）
    if self.args.acc_info == "on":
        out_loss = [["epoch",'train_loss','test_loss', 'train_acc', 'test_acc']]
        for idx, (train_loss, test_loss, train_acc, test_acc ) in enumerate(zip(train_loss, test_loss, train_acc, test_acc)):
            out_loss.append([idx+1,train_loss, test_loss, train_acc, test_acc])
        out_loss.append([elapsed_time[0]])
    else:
        out_loss = [["epoch",'train_loss','test_loss', 'train_acc', 'test_acc']]
        for idx, (train_loss, test_loss) in enumerate(zip(train_loss, test_loss)):
            out_loss.append([idx+1,train_loss, test_loss])
        out_loss.append([elapsed_time[0]])
    

    model = self
    summary = []
    summary.extend([out_loss, model, out_pre_w, out_post_w, shape_box, epoch_num])
    return summary