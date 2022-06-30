from collections import Counter
import numpy as np 
import json
import os

def provider_partition(r,news_index,provider_dict,news_provider_index,data_root_path):
    
    with open(os.path.join(data_root_path,'news_click_release.json'),'rb') as f:
        news_click = json.load(f)

    provider_click = np.zeros((len(provider_dict)+1,))

    for nid in news_click:
        nix = news_index[nid]
        pix = news_provider_index[nix]
        provider_click[pix] += news_click[nid]

    news_num = np.zeros((len(provider_dict)+1,))
    for i in range(len(provider_dict)+1):
        news_num[i] = ((news_provider_index==i)).sum()
    
    
    num = int(len(provider_dict)*r)
    protected_group = set((provider_click/(news_num+1000)).argsort()[:num])
    news_ratio = 0
    for pix in protected_group:
        news_ratio += (news_provider_index==pix).sum()
    news_ratio /= len(news_index)+1
    
    return protected_group, news_ratio


def ER(Prob,ratio):
    r1 = Prob/ratio
    r2 = (1-Prob)/(1-ratio)
    return r1/r2

def log2(x):
    return np.log(x)/np.log(2)

def rND(Prob,ratio,TOPK):
    seq = np.abs(Prob-ratio)
    res = [0]
    Z = [0]
    for i in range(len(TOPK)):
        r = res[-1] + seq[i]/log2(TOPK[i])
        res.append(r)
        r = Z[-1] + ratio/log2(TOPK[i])
        Z.append(r)
    res = res[1:]
    Z = Z[1:]
    return np.array(res)/np.array(Z)


def fairness_evaluation(news_scoring,test_user_scoring,protected_group, news_ratio,news_provider_index):
    TOPK = [10,30,50,100,150,200]
    Prob = np.zeros((len(TOPK),))
    
    ITERATION = 50000
    
    for uid in range(ITERATION):
        uv = test_user_scoring[uid]    
        scores = np.dot(news_scoring,uv)
        inx = (-scores).argsort()[:TOPK[-1]]
        
        for ix_k in range(len(TOPK)):
            K = TOPK[ix_k]
            pro_inx = news_provider_index[inx[:K]]
            pro_inx = Counter(pro_inx)
            for pid, ct in pro_inx.items():
                if not pid in protected_group:
                    continue
                Prob[ix_k] += ct/K
                
    Prob /= ITERATION
    score_ER = ER(Prob,news_ratio)
    score_rND = rND(Prob,news_ratio,TOPK)
    
    
    return score_ER,score_rND


def dump_performance(test_imps,news_scoring,user_scoring,):
    result = []
    for i in range(len(test_imps)):
        
        nids = test_imps[i]['docs']

        uv = user_scoring[i]
        nvs = news_scoring[nids]
        scores = np.dot(nvs,uv)
        scores = (-scores).argsort()
        rank = np.zeros((len(scores),),dtype='int32')
        for j in range(len(scores)):
            rank[scores[j]] = j+1
        rank = json.dumps(rank.tolist())
        rank = ''.join(rank.split(' '))
        line = str(i+1) + ' '+ rank + '\n'
        result.append(line)
        
    with open('prediction.txt','w') as f:
        for i in range(len(result)):
            f.write(result[i])