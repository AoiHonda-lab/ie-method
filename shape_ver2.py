# -*- coding: utf-8; -*-

from itertools import combinations
import calc

#w:重み syugou:args.add omega:変数の数
def get_shape(w, term_num, args):
    # w:空集合を含めた包除積分の重み（メビウス変換の値）
    # term_num:データの項数、変数の数
    # args:args.addとargs.matrixtypeの二つのint値を使って条件分岐
    mob_fuzy = w
    w_add = args.add
    def daisu(ie_data_len, add): #args.addで組み合わせの調整を行う。dais(3,2)としたら[(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3)]
    # 代数積を取得
        items = [i for i in range(1, ie_data_len+1)]
        subsets=[]
        for i in range(len(items) + 1):
            if i > add:#二加法的まで
                break
            for c in combinations(items, i):
                subsets.append(c)
                # subsets.append(list(c))
        hh = subsets
        return hh 

    # 部分集合取得 2,3の場合は包除積分の特殊系の集合を取得する
    if args.matrixtype == 2:
        all_syugou = calc.rnn_matrix_tuple(term_num)
    elif args.matrixtype == 3:
        all_syugou = calc.bi_rnn_matrix_tuple(term_num)
    else:
        all_syugou = daisu(term_num, w_add)

    d_mob = dict(zip(all_syugou, mob_fuzy)) #辞書化して各々対応した要素に重みを入れている
    l = list(d_mob) #list化した空集合を含む集合
    shap_sum_i = []
    for i in range(1, len(l)):
        # key_a = l[j]
        shap = []
        if i == 1:
            search = 1
        elif len(l[i])-len(l[i-1]) == 1:
            search = i
        else:
            pass
        for j in range(search, len(l)):
            T = l[i]
            if  set(T) <= set(l[j]):
                w = 1/(len(l[j])-len(T)+1) * d_mob[l[j]] #(13)の式
                shap.append(w)
        shap_sum_i.append(sum(shap))
        

    return shap_sum_i