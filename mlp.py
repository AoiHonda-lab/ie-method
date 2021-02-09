from __future__ import division

import numpy as np
import pandas as pd
import math
import chainer
import chainer.links as L
import chainer.functions as F
# import relu_1
from chainer.dataset import concat_examples
from chainer.cuda import to_cpu
from chainer import training
from chainer import Variable
from chainer import optimizers
import copy
import time
# import chainerx


class MLP(chainer.Chain):

    def __init__(self, args):
        super(MLP, self).__init__()
        # super(MLP, self).__init__(initialW = None, initial_bias = None)
        self.args = args
        with self.init_scope():
            self.fc1 = L.Linear(None, args.mlp_units)   
            self.fc2 = L.Linear(args.mlp_units, args.mlp_units)  
            self.fc3 = L.Linear(args.mlp_units, args.mlp_units)       
            self.fc4 = L.Linear(args.mlp_units, self.args.out)

    # def relu_v(self, inputs):
    #     x = inputs
    #     self.y = 1 * x
    #     self.y = np.where(self.y.data < 0, 0, np.where(self.y.data > 1, 1, self.y.data))
    #     return Variable(self.y)

    def __call__(self, x):
        # h1 = self.relu_v(self.fc1(x))
        # h1 = self.relu_v(self.fc1(x))
        h1 = F.sigmoid(self.fc1(x))
        h2 = F.sigmoid(self.fc2(h1))
        h3 = F.sigmoid(self.fc3(h2))
        # h1 = F.relu(self.fc1(x))
        # h1_= Variable(np.where(h1.data > 1, 1, h1.data))
        # h2 = self.relu_v(self.fc2(h1))
        # h2 = F.relu(self.fc2(h1))
        # h2_ =  Variable(np.where(h2.data > 1, 1, h2.data))
        y = self.fc4(h3)
        if self.args.lossf == "mse_sig":
            return F.sigmoid(y)
        else:
            pass
        return y


    def train_model(self, train_iter, test_iter, optimizer, elapsed_time, start, args):
        

        # train
        train_loss = []
        test_loss = []
        train_acc = []
        test_acc = []

        test_loss_point = 10000000000000000000

        if self.args.gpu_id >= 0:
            self.to_gpu(self.args.gpu_id)

        print("===== read epoch =====", flush=True)
        for epoch in range(self.args.epoch):
            #print("epoch: {}".format(epoch), end ="")
            err_temp = 0
            acc_temp = 0
    
            for i in range(0, len(train_iter.dataset), self.args.batch_size): # batch size loop                
                train_batch = train_iter.next()
                x, target = concat_examples(train_batch, self.args.gpu_id)
                if args.data == "car":
                    target = F.reshape(target,(len(train_batch), -1))
                y = self(x)
                if i == 0 and epoch == 0:
                    inti_model = copy.deepcopy(self)
                if args.data == "car":
                    loss = F.mean_squared_error(y, target)
                elif args.data == "mnist":
                    # 出力１の時
                    if self.args.lossf == "ent":
                        loss = F.sigmoid_cross_entropy(y,F.reshape(target,(self.args.batch_size,1)))
                        y_ = F.concat((1-F.sigmoid(y), F.sigmoid(y)),axis=1)
                        acc = F.accuracy(y_, target.astype('int32'))
                    elif self.args.lossf == "mse_sig":
                        loss  = F.mean_squared_error(y, F.reshape(target,(self.args.batch_size,1)).data.astype(np.float32))
                        y_ = F.concat((1-y, y),axis=1)
                        acc = F.accuracy(y_.array, target.astype('int32'))
                    elif self.args.lossf == "mse":
                        loss  = F.mean_squared_error(y, F.reshape(target,(self.args.batch_size,1)).data.astype(np.float32))
                        y_ = [1 if r > 0.5 else 0 for r in y.data]
                        tf = np.array(y_) == np.array(list(target))
                        acc = Variable(np.array(np.count_nonzero(tf) / self.args.batch_size))
                    # loss = F.softmax_cross_entropy(y, target)
                    # acc = F.accuracy(y, target)
                    # if self.args.lossf == "ent":
                    #loss = F.mean_squared_error(y,F.reshape(target,(self.args.batch_size,1)).data.astype(np.float32))
                    #loss = F.sigmoid_cross_entropy(y,F.reshape(target,(self.args.batch_size,1)))
                    #y_ = F.concat((1-F.sigmoid(y), F.sigmoid(y)),axis=1)
                    #acc = F.accuracy(y_, target)

                self.cleargrads()
                loss.backward()
                optimizer.update()
                err_temp += float(loss.data)
                acc_temp += float(acc.data)
                # err_temp_ += float(loss.data)

            length = math.ceil(len(train_iter.dataset)/self.args.batch_size)
            train_loss.append(err_temp/length)
            train_acc.append(acc_temp/length)
            if self.args.acc_info == "on":
                print("epoch: {} train_loss: {:.4f} train_acc: {:.4f}".format(epoch, err_temp/length, acc_temp/length), end ="")
            else:
                print("epoch: {} train_loss: {:.4f}".format(epoch, err_temp/length), end ="")
                #print(" train_loss: {:.4f}".format(err_temp/len(train_iter.dataset)), end ="")

            test_batch = test_iter.next()
            x, target = concat_examples(test_batch, self.args.gpu_id)
            if args.data == "car":
                target = F.reshape(target,(len(train_batch), -1))
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                y = self(x)
            
            if args.data == "car":
                loss = F.mean_squared_error(y, target)
            elif args.data == "mnist":
                # 出力１の時
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
                # loss = F.softmax_cross_entropy(y, target)
                # acc = acc = F.accuracy(y, target)
                #loss = F.mean_squared_error(y,F.reshape(target,(len(test_batch),1)).data.astype(np.float32))
                #loss = F.sigmoid_cross_entropy(y,F.reshape(target,(len(test_batch),1)))
                #y_ = F.concat((1-F.sigmoid(y), F.sigmoid(y)),axis=1)
                #acc = F.accuracy(y_, target)
            test_loss.append(float(to_cpu(loss.data)))
            test_acc.append(float(to_cpu(acc.data)))
            if self.args.acc_info == "on":
                print(" test_loss: {:.4f} test_acc: {:.4f}".format(float(to_cpu(loss.data)), float(to_cpu(acc.data))))
            else:
                print(" test_loss: {:.4f}".format(float(to_cpu(loss.data))))
           
            #print(" test_loss: {:.4f}".format(float(to_cpu(loss.data)))) #test_acc: {} , float(to_cpu(acc.data))

            test_iter.reset()

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
            last_train = float(to_cpu(acc.data))#acc_temp/len(train_iter.dataset) 
            
            if last_train > args.limit:
                max_epoch = epoch
                break

        

        test_last_loss = []
        test_last_acc = []
        # test_batch = test_iter.next()
        # x, target = concat_examples(test_batch, self.args.gpu_id)
        # if args.data == "car":
        #     target = F.reshape(target,(len(train_batch), -1))
        
        # with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        #     y = self(x)
        # if args.data == "car":
        #     loss = F.mean_squared_error(y, target)
        # elif args.data == "mnist":
        #     # 出力１の時
        #     if self.args.lossf == "ent":
        #         loss = F.sigmoid_cross_entropy(y,F.reshape(target,(self.args.batch_size,1)))
        #         y_ = F.concat((1-F.sigmoid(y), F.sigmoid(y)),axis=1)
        #         acc = F.accuracy(y_, target.astype('int32'))
        #     elif self.args.lossf == "mse_sig":
        #         loss  = F.mean_squared_error(F.sigmoid(y), F.reshape(target,(self.args.batch_size,1)).data.astype(np.float32))
        #         y_ = F.concat((1-F.sigmoid(y), F.sigmoid(y)),axis=1)
        #         acc = F.accuracy(y_.array, target.astype('int32'))
        #     elif self.args.lossf == "mse":
        #         loss  = F.mean_squared_error(y, F.reshape(target,(self.args.batch_size,1)).data.astype(np.float32))
        #         y_ = [1 if r > 0.5 else 0 for r in y.data]
        #         tf = np.array(y_) == np.array(list(target))
        #         acc = Variable(np.array(np.count_nonzero(tf) / self.args.batch_size))
            # loss = F.softmax_cross_entropy(y, target)
            # acc = acc = F.accuracy(y, target)
            #loss = F.mean_squared_error(y,F.reshape(target,(len(test_batch),1)).data.astype(np.float32))
            #loss = F.sigmoid_cross_entropy(y,F.reshape(target,(len(test_batch),1)))
            #y_ = F.concat((1-F.sigmoid(y), F.sigmoid(y)),axis=1)
            #acc = F.accuracy(y_, target)

        #test_last_loss.append(float(to_cpu(loss.data)))
        #test_last_acc.append(float(to_cpu(acc.data)))
        
        #test_iter.reset()
        elapsed_time.append(time.time() - start)
        print("===== read time =====", flush=True)
        print('time:{}'.format(elapsed_time[0]))
        # train_loss, test_loss, w_b = model.train_model(train_iter, test_iter, optimizer)
        
        # 学習結果をまとめる
        out_loss = [["epoch",'train_loss','test_loss', 'train_acc', 'test_acc']]
        for idx, (train_loss, test_loss, train_acc, test_acc ) in enumerate(zip(train_loss, test_loss, train_acc, test_acc)):
            out_loss.append([idx+1,train_loss, test_loss, train_acc, test_acc])
        out_loss.append([elapsed_time[0]])

        model = self
        summary = []
        summary.extend([out_loss, model])
        return summary

        
        # w_last = np.array(self.lt.W.data).T
        # w_first = []
        # w_first.append(self.l1.W.data)
        # w_first.append(self.l2.W.data)
        # w_first.append(self.l3.W.data)
        # w_first.append(self.l4.W.data)
        # w_first.append(self.l5.W.data)
        # w_first.append(self.l6.W.data)
        # w_first_ = np.array(w_first).T
        


        #return train_loss, test_loss, test_last_loss, train_acc, test_acc, test_last_acc, max_epoch
        # return train_loss, test_test, pd_w_b