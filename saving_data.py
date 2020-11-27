# -*- coding: utf-8; -*-

# library
import argparse
import pickle
import numpy as np
import csv

# saving result data
def saving_ie(summry, args, no = 0):

	"""
	summry --> ([out_loss, model, out_pre_w, out_post_w, shape])
	"""

	# 損失関数と正解率をエクセルデータに保存
	with open('./result/train/value/result_shape{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_monotony_{}_0_{}_mno{}.csv'.format(args.day, args.norm, args.l_lr, args.shoki_opt, args.out, args.lossf, args.add, args.func, args.opt, args.lr, args.pre_shoki, args.data_model, args.model, args.not_monotony ,args.train_number, no), 'w') as f:
		writer = csv.writer(f, lineterminator='\n', delimiter='\t')
		writer.writerows(summry[0])


	# モデルをpklに保存
	summry[1].to_cpu()
	pickle.dump(summry[1], open("./result/train/pkl/model_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_monotony_{}_{}_0_{}_mno{}.pkl".format(args.day, args.norm, args.l_lr, args.shoki_opt, args.out, args.lossf, args.add, args.func, args.opt, args.lr ,args.pre_shoki, args.data_model, args.model, args.not_monotony, args.train_number, no), "wb"), -1)


	# 重みデータをpklに保存
	w_model = []
	w_model.append(summry[2])
	w_model.append(summry[3])
	pickle.dump(w_model, open("./result/train/ww/w_model/W_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_monotony_{}_{}_0_{}_mno{}.pkl".format(args.day, args.norm, args.l_lr, args.shoki_opt, args.out, args.lossf, args.add, args.func, args.opt, args.lr ,args.pre_shoki, args.data_model, args.model, args.not_monotony, args.train_number, no), "wb"), -1)
	
	
	# 全重みをエクセルデータに保存
	pre = np.array(summry[2]).T
	post = np.array(summry[3]).T
	if args.pre_ie == "precor":
		out_w = np.hstack([pre,post])
	else:
		out_w = post

	with open("./result/train/ww/w_ww/W_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_monotony_{}_{}_0_{}_mno{}.csv".format(args.day, args.norm, args.l_lr, args.shoki_opt, args.out, args.lossf, args.add, args.func, args.opt, args.lr ,args.pre_shoki, args.data_model, args.model, args.not_monotony, args.train_number, no), 'w') as f:
		writer = csv.writer(f, lineterminator='\n', delimiter='\t')
		writer.writerows(out_w)

	# シャプレイ値保存
	np.savetxt('./result/train/shape/shape_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_monotony_{}_0_{}_mno{}.csv'.format(args.day, args.norm, args.l_lr, args.shoki_opt, args.out, args.lossf, args.add, args.func, args.opt, args.lr, args.pre_shoki, args.data_model, args.model, args.not_monotony ,args.train_number, no), np.array(summry[4]) ,fmt='%.4f',delimiter=',')
	# np.savetxt("shape_{}.csv".format(filename), shape_box ,fmt='%.4f',delimiter=',')


def saving_mlp(summry, args, no = 0):

	"""
	summry --> ([out_loss, model, out_pre_w, out_post_w, shape])
	"""

	# 損失関数と正解率をエクセルデータに保存
	with open('./result/train/value/result_shape{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_units{}_mno{}.csv'.format(args.day, args.norm, args.l_lr, args.shoki_opt, args.out, args.lossf,  args.func, args.opt, args.lr, args.data_model, args.model,args.train_number, args.mlp_units, no), 'w') as f:
		writer = csv.writer(f, lineterminator='\n', delimiter='\t')
		writer.writerows(summry[0])
