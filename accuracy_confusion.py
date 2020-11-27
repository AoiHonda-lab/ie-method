from sklearn.metrics import confusion_matrix
import numpy as np

def accuracy(args,model,data,y):
    test_suvive = []
	test_ID = []
	if args.data_model == "artificial":
		pass
	elif args.data_model == "Titanic_all_1_class":
		for i in range(len(model(Rtest._datasets[0]))):
			test_ID.append(int(892+i))
			if model(Rtest._datasets[0])[i].data[0] > 1/2:
				test_suvive.append(1)
			else:
				test_suvive.append(0)
		if args.model =="ie":
			np.savetxt('./result/test/modeltest_{}_{}{}_{}_epoch{}_{}_{}_{}_DefValu-{}_monotony-{}.csv'.format(args.day,args.model,args.add,args.func,args.epoch,args.lossf,args.opt,args.data_model,args.not_ie_shoki,args.not_monotony),np.array([test_ID,test_suvive],dtype = 'int32').T, header='PassengerId,Survived',fmt='%d',delimiter=',',comments='')
		else:
			np.savetxt('./result/test/modeltest_{}_{}{}_{}_epoch{}_{}_{}_{}.csv'.format(args.day,args.model,args.mid_l1,args.func,args.epoch,args.lossf,args.opt,args.data_model),np.array([test_ID,test_suvive],dtype = 'int32').T, header='PassengerId,Survived',fmt='%d',delimiter=',',comments='')
	elif args.data_model == "Titanic_normal_1_class" and args.bagging > 1:
		for i in range(args.bagging):
			with open("./result/train/pkl/model_{}_{}{}_epoch{}_out{}_{}_{}_{}_{}_{}_shokiti{}_bg{}_monotony_{}.pkl".format(args.day,args.model, args.add, args.epoch, args.out, args.lossf, args.func, args.opt, args.lr , args.data_model,args.not_ie_shoki,i,args.not_monotony), mode='rb') as f:
				model = pickle.load(f)
			if i == 0:
				bagging_predict = model(Rtest._datasets[0])
			else:
				bagging_predict += model(Rtest._datasets[0])
		for i in range(len(model(Rtest._datasets[0]))):
			test_ID.append(int(892+i))
			if bagging_predict[i].data[0]/args.bagging > 1/2:
				test_suvive.append(1)
			else:
				test_suvive.append(0)	
		if args.model =="ie":
			np.savetxt('./result/test/modeltest_{}_{}{}_{}_epoch{}_{}_{}_{}_DefValu-{}_monotony-{}.csv'.format(args.day,args.model,args.add,args.func,args.epoch,args.lossf,args.opt,args.data_model,args.not_ie_shoki,args.not_monotony),np.array([test_ID,test_suvive],dtype = 'int32').T, header='PassengerId,Survived',fmt='%d',delimiter=',',comments='')
		else:
			np.savetxt('./result/test/modeltest_{}_{}{}_{}_epoch{}_{}_{}_{}.csv'.format(args.day,args.model,args.mid_l1,args.func,args.epoch,args.lossf,args.opt,args.data_model),np.array([test_ID,test_suvive],dtype = 'int32').T, header='PassengerId,Survived',fmt='%d',delimiter=',',comments='')
	else:
		for i in range(len(model(Rtest._datasets[0]))):
			test_ID.append(int(892+i))
			if model(Rtest._datasets[0])[i].data[0] > 1/2:
				test_suvive.append(1)
			else:
				test_suvive.append(0)
		if args.model =="ie":
			np.savetxt('./result/test/modeltest_{}_{}{}_{}_epoch{}_{}_{}_{}_DefValu-{}_monotony-{}.csv'.format(args.day,args.model,args.add,args.func,args.epoch,args.lossf,args.opt,args.data_model,args.not_ie_shoki,args.not_monotony),np.array([test_ID,test_suvive],dtype = 'int32').T, header='PassengerId,Survived',fmt='%d',delimiter=',',comments='')
		else:
			np.savetxt('./result/test/modeltest_{}_{}{}_{}_epoch{}_{}_{}_{}.csv'.format(args.day,args.model,args.mid_l1,args.func,args.epoch,args.lossf,args.opt,args.data_model),np.array([test_ID,test_suvive],dtype = 'int32').T, header='PassengerId,Survived',fmt='%d',delimiter=',',comments='')
	