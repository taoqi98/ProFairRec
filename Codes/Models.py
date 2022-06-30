import numpy as np
import keras
from keras.layers import Embedding
from keras.layers import *
from keras import backend as K
from keras.optimizers import *
from keras.models import Model
from keras.utils import multi_gpu_model

from Hypers import *


class Attention(Layer):
 
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Attention, self).__init__(**kwargs)
 
    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)
 
    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12
 
    def call(self, x):
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x

        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))

        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))
        A = K.softmax(A)

        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq
 
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


def AttentivePooling(dim1,dim2):
    vecs_input = Input(shape=(dim1,dim2),dtype='float32')
    user_vecs =Dropout(0.2)(vecs_input)
    user_att = Dense(200,activation='tanh')(user_vecs)
    user_att = keras.layers.Flatten()(Dense(1)(user_att))
    user_att = Activation('softmax')(user_att)
    user_vec = keras.layers.Dot((1,1))([user_vecs,user_att])
    model = Model(vecs_input,user_vec)
    return model

def get_doc_encoder(word_embedding_matrix):

    sentence_input = Input(shape=(MAX_SENTENCE,),dtype='int32')
    
    word_embedding_layer = Embedding(word_embedding_matrix.shape[0], 300, weights=[word_embedding_matrix],trainable=True)
    word_vecs = word_embedding_layer(sentence_input)
    droped_vecs = Dropout(0.2)(word_vecs)
    
    word_rep = Attention(20,20)([droped_vecs]*3)
    #word_rep = Activation('relu')(word_rep)
    
    droped_rep = Dropout(0.2)(word_rep)    
    title_vec = AttentivePooling(MAX_SENTENCE,400)(droped_rep)

    sentEncodert = Model(sentence_input, title_vec)
    
    return sentEncodert


def get_user_encoder():
    user_vecs_input = Input(shape=(MAX_ALL,400))    
    click_mask_input = Input(shape=(MAX_ALL,))
    
    user_vecs = Dropout(0.2)(user_vecs_input)

    user_vecs = Attention(20,20)([user_vecs]*3)
    user_vec = AttentivePooling(MAX_ALL,400)(user_vecs)
        
    return Model(user_vecs_input,user_vec)

def ConsineLayer():
    vec_input = Input(shape=(800,))
    vec1 = Lambda(lambda x:x[:,:400])(vec_input)
    vec2 = Lambda(lambda x:x[:,400:])(vec_input)
    score = Dot(axes=-1,normalize=True)([vec1,vec2])
    return Model(vec_input,score)

def get_classfier(CT):
    vec_input = Input((400,))
    logit = Dense(CT,activation='softmax')(vec_input)
    model = Model(vec_input,logit)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0001), 
                  #optimizer= SGD(lr=0.01),
                  metrics=['acc'])
    return model

def create_model(CT,title_word_embedding_matrix,provider_dict,news_hyper=1, user_hyper=1, adv_hyper=1/250):
        
    news_encoder = get_doc_encoder(title_word_embedding_matrix)
    user_encoder = get_user_encoder()
    user_encoder2 = get_user_encoder()

    clicked_title_input = Input(shape=(MAX_ALL,MAX_SENTENCE,), dtype='int32')    
    title_inputs = Input(shape=(1+npratio,MAX_SENTENCE,),dtype='int32') 
    
    clicked_provider_input = Input(shape=(MAX_ALL,), dtype='int32')    
    provider_inputs = Input(shape=(1+npratio,),dtype='int32') 

    attack_inputs = Input(shape=(MAX_SENTENCE,),dtype='int32') 
    
    user_vecs = TimeDistributed(news_encoder)(clicked_title_input)
    news_vecs = TimeDistributed(news_encoder)(title_inputs)

    provider_embedding_layer = Embedding(len(provider_dict)+1, 400, trainable=True)
    
    clicked_provider_vecs = provider_embedding_layer(clicked_provider_input)
    provider_vecs = provider_embedding_layer(provider_inputs) 
   
    user_vecs2 = clicked_provider_vecs
    news_vecs2 = provider_vecs
    

    user_vec = user_encoder(user_vecs)
    user_vec2 = user_encoder2(user_vecs2)
    
    all_content = Concatenate(axis=-2)([user_vecs,news_vecs])
    
    classfier = get_classfier(CT)
    classfier.trainable = False
    
    attack_vec = news_encoder(attack_inputs)
    attack_logits = classfier(attack_vec)
    
    user_vec = Lambda(lambda x:(x[0]+x[1])/2)([user_vec,user_vec2])
    news_vecs = Lambda(lambda x:(x[0]+x[1])/2)([news_vecs,news_vecs2])

    news_vecs = Dropout(0.2)(news_vecs)
    

    scores = keras.layers.Dot(axes=-1)([news_vecs,user_vec])
    
    cosine_layer = ConsineLayer()
    user_regu_vec = Concatenate(axis=-1)([user_vec,user_vec2])
    user_regu_loss = cosine_layer(user_regu_vec)
    user_regu_loss = Lambda(lambda x:x/50)(user_regu_loss)
    user_regu_loss = Lambda(lambda x:user_hyper*x)(user_regu_loss)
    
    all_content_vecs = Concatenate(axis=-2)([user_vecs,news_vecs])
    all_provider_vecs = Concatenate(axis=-2)([clicked_provider_vecs,provider_vecs]) #(52,400)
    
    all_regu_vecs = Concatenate(axis=-1)([all_content_vecs,all_provider_vecs])
    regu_loss = TimeDistributed(cosine_layer)(all_regu_vecs)

    regu_loss = Lambda(lambda x:K.mean(x,axis=-2))(regu_loss)

    regu_loss = Reshape((1,))(regu_loss)
    regu_loss = Lambda(lambda x:news_hyper*x)(regu_loss)
    
    regu_loss = Add()([regu_loss,user_regu_loss])
    

    logits = keras.layers.Activation(keras.activations.softmax,name = 'fair')(scores)     


    model = Model([title_inputs, provider_inputs, clicked_title_input,clicked_provider_input,attack_inputs],[logits,regu_loss,attack_logits]) # max prob_click_positive
    model.compile(loss=['categorical_crossentropy','mean_absolute_error','categorical_crossentropy'],
                  optimizer=Adam(lr=0.0001), 
                  loss_weights=[1,1,adv_hyper],
                  metrics=['acc'])

    return model,news_encoder,user_encoder,user_encoder2,provider_embedding_layer,classfier