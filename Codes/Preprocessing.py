from Hypers import *
from Utils import *
from nltk.tokenize import word_tokenize
import json

def read_news(path,filenames):
    news={}
    category=[]
    subcategory=[]
    news_index={}
    index=1
    word_dict={}
    word_index=1
    with open(os.path.join(path,filenames)) as f:
        lines=f.readlines()
    for line in lines:
        splited = line.strip('\n').split('\t')
        doc_id,vert,subvert,title= splited[0:4]
        news_index[doc_id]=index
        index+=1
        category.append(vert)
        subcategory.append(subvert)
        title = title.lower()
        title=word_tokenize(title)
        news[doc_id]=[vert,subvert,title]
        for word in title:
            word = word.lower()
            if not(word in word_dict):
                word_dict[word]=word_index
                word_index+=1
    category=list(set(category))
    subcategory=list(set(subcategory))
    category_dict={}
    index=1
    for c in category:
        category_dict[c]=index
        index+=1
    subcategory_dict={}
    index=1
    for c in subcategory:
        subcategory_dict[c]=index
        index+=1
    return news,news_index,category_dict,subcategory_dict,word_dict


def get_doc_input(news,news_index,category,subcategory,word_dict):
    news_num=len(news)+1
    news_title=np.zeros((news_num,MAX_SENTENCE),dtype='int32')
    news_vert=np.zeros((news_num,),dtype='int32')
    news_subvert=np.zeros((news_num,),dtype='int32')
    for key in news:    
        vert,subvert,title=news[key]
        doc_index=news_index[key]
        news_vert[doc_index]=category[vert]
        news_subvert[doc_index]=subcategory[subvert]
        for word_id in range(min(MAX_SENTENCE,len(title))):
            news_title[doc_index,word_id]=word_dict[title[word_id].lower()]
        
    return news_title,news_vert,news_subvert

def load_provider_data(news_index,data_root_path):

    news_provider = {}
    news_provider_index = np.zeros((len(news_index)+1,),dtype='int32')
    provider_dict = {}
    provider_index = 1
    with open(os.path.join(data_root_path,'news_provider.json')) as f:
        data = json.load(f)

    for nid in data:
        url = data[nid]
        if url == '':
            news_provider[nid] = url
            continue
        if not url in provider_dict:
            provider_dict[url] = provider_index
            provider_index += 1
            
        news_provider_index[news_index[nid]] = provider_dict[url]
    
    return news_provider,news_provider_index,provider_dict


def read_clickhistory(news_index,data_root_path,filename):
    
    lines = []
    userids = []
    with open(os.path.join(data_root_path,filename)) as f:
        lines = f.readlines()
        
    sessions = []
    for i in range(len(lines)):
        _,uid,eventime, click, imps = lines[i].strip().split('\t')
        if click == '':
            clicks = []
        else:
            clikcs = click.split()
        true_click = []
        for click in clikcs:
            if not click in news_index:
                continue
            true_click.append(click)
        pos = []
        neg = []
        for imp in imps.split():
            docid, label = imp.split('-')
            if label == '1':
                pos.append(docid)
            else:
                neg.append(docid)
        sessions.append([true_click,pos,neg])
    return sessions

def read_part_clickhistory(news_index,data_root_path,filename,num=200000):
    
    lines = []
    userids = []
    with open(os.path.join(data_root_path,filename)) as f:
        for i in range(num):
            l = f.readline()
            lines.append(l)
        
    sessions = []
    for i in range(len(lines)):
        _,uid,eventime, click, imps = lines[i].strip().split('\t')
        if click == '':
            clicks = []
        else:
            clikcs = click.split()
        true_click = []
        for click in clikcs:
            if not click in news_index:
                continue
            true_click.append(click)
        pos = []
        neg = []
        for imp in imps.split():
            docid, label = imp.split('-')
            if label == '1':
                pos.append(docid)
            else:
                neg.append(docid)
        sessions.append([true_click,pos,neg])
    return sessions

def parse_user(news_index,session):
    user_num = len(session)
    user={'click': np.zeros((user_num,MAX_ALL),dtype='int32'),}
    for user_id in range(len(session)):
        tclick = []
        click, pos, neg =session[user_id]
        for i in range(len(click)):
            tclick.append(news_index[click[i]])
        click = tclick

        if len(click) >MAX_ALL:
            click = click[-MAX_ALL:]
        else:
            click=[0]*(MAX_ALL-len(click)) + click
            
        user['click'][user_id] = np.array(click)
    return user


def get_train_input(news_index,session):
    sess_pos = []
    sess_neg = []
    user_id = []
    for sess_id in range(len(session)):
        sess = session[sess_id]
        _, poss, negs=sess
        for i in range(len(poss)):
            pos = poss[i]
            neg=newsample(negs,npratio)
            sess_pos.append(pos)
            sess_neg.append(neg)
            user_id.append(sess_id)

    sess_all = np.zeros((len(sess_pos),1+npratio),dtype='int32')
    label = np.zeros((len(sess_pos),1+npratio))
    for sess_id in range(sess_all.shape[0]):
        pos = sess_pos[sess_id]
        negs = sess_neg[sess_id]
        sess_all[sess_id,0] = news_index[pos]
        index = 1
        for neg in negs:
            sess_all[sess_id,index] = news_index[neg]
            index+=1
        #index = np.random.randint(1+npratio)
        label[sess_id,0]=1
    user_id = np.array(user_id, dtype='int32')
    
    return sess_all, user_id, label

def get_test_input(news_index,session):
    
    Impressions = []
    userid = []
    for sess_id in range(len(session)):
        _, poss, negs = session[sess_id]
        imp = {'labels':[],
                'docs':[]}
        userid.append(sess_id)
        for i in range(len(poss)):
            docid = news_index[poss[i]]
            imp['docs'].append(docid)
            imp['labels'].append(1)
        for i in range(len(negs)):
            docid = news_index[negs[i]]
            imp['docs'].append(docid)
            imp['labels'].append(0)
        Impressions.append(imp)
        
    userid = np.array(userid,dtype='int32')
    
    return Impressions, userid,

def build_adversarial_label(news_index,provider_dict,news_provider_index):
    news_provider_ct = np.zeros((len(provider_dict)+1,))
    for pix in range(len(provider_dict)+1):
        news_provider_ct[pix] += (news_provider_index==pix).sum()
    CT = (news_provider_ct>100).sum() + 1
    CT = int(CT)

    Truncated_Provider_Dict = {}
    pcount = 1
    for pix in range(len(provider_dict)+1):
        if news_provider_ct[pix]<=100:
            Truncated_Provider_Dict[pix] = 0
        else:
            Truncated_Provider_Dict[pix] = pcount
            pcount += 1

    news_attack_label = np.zeros((len(news_index)+1,CT))
    for nix in range(len(news_index)+1):
        pix = news_provider_index[nix]
        pid = Truncated_Provider_Dict[pix]
        news_attack_label[nix,pid] = -1
    
    return news_attack_label,CT


def read_test_clickhistory_noclk(news_index,data_root_path,filename):
    
    lines = []
    with open(os.path.join(data_root_path,filename)) as f:
        lines = f.readlines()
    sessions = []
    for i in range(len(lines)):
        _,uid,eventime, click, imps = lines[i].strip().split('\t')
        if click == '':
            clicks = []
        else:
            clicks = click.split()
        true_click = []
        for j in range(len(clicks)):
            click = clicks[j]
            assert click in news_index
            true_click.append(click)
        pos = []
        neg = []
        for imp in imps.split():
            pos.append(imp)
        sessions.append([true_click,pos,neg])
    return sessions
