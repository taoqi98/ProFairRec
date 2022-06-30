import numpy as np
from keras.utils import Sequence
from keras.utils import Sequence

class get_hir_train_generator(Sequence):
    def __init__(self,news_title, provider, clicked_news,user_id, news_id, label,news_attack_label, batch_size):
        self.title = news_title
        self.provider = provider
        
        self.clicked_news = clicked_news

        self.user_id = user_id
        self.doc_id = news_id
        self.label = label
        
        self.label2 = np.zeros((batch_size,1))
        self.news_attack_label = news_attack_label
        
        self.batch_size = batch_size
        self.ImpNum = self.label.shape[0]
        
    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))
    
    def __get_news(self,docids):
        title = self.title[docids]
        provider = self.provider[docids]
        
        return [title,provider]
        

    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed> self.ImpNum:
            ed = self.ImpNum
        
        doc_ids = self.doc_id[start:ed]
        title = self.__get_news(doc_ids)
        
        user_ids = self.user_id[start:ed]
        clicked_ids = self.clicked_news[user_ids]
        
        user_title = self.__get_news(clicked_ids)
        
        label = self.label[start:ed]
        label2 = self.label2[:ed-start]
                

        att_ids = np.random.randint(1,self.news_attack_label.shape[0],(ed-start,))
        att_input = self.title[att_ids]
        attack_labels = self.news_attack_label[att_ids]
        
        max_game_x = self.title[att_ids]

        max_game_y = -self.news_attack_label[att_ids]
                
        return  ([max_game_x,max_game_y], title+ user_title + [att_input],[label,label2,attack_labels])

class get_hir_user_generator(Sequence):
    def __init__(self,news_title, clicked_news,batch_size):
        self.title = news_title
        self.clicked_news = clicked_news

        self.batch_size = batch_size
        self.ImpNum = self.clicked_news.shape[0]
        
    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))

    def __get_news(self,docids):
        title = self.title[docids]
        
        return title
            
    
    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed> self.ImpNum:
            ed = self.ImpNum
        clicked_ids = self.clicked_news[start:ed]
        
        user_title = self.__get_news(clicked_ids)

        return user_title