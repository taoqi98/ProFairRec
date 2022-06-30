from Hypers import *
import numpy as np

def AVG(arr):
    m = np.array(arr).mean()
    m = m*10000
    m = int(m)
    m = m/10000
    return m

def model_training(model,news_encoder,classfier,train_generator,EPOCH):
    all_rec_loss = []
    all_rec_acc = []
    all_att_loss = []
    all_att_acc = []
    all_dis_loss = []
    all_dis_acc = []
    all_sum_loss = []


    for epoch in range(EPOCH):
        count = 0
        for att_sam,x,y in train_generator:
            max_game_x, max_game_y= att_sam
            max_game_x = news_encoder.predict(max_game_x,verbose=False)

            classfier.trainable = True
            dis_loss,dis_acc = classfier.train_on_batch(max_game_x,max_game_y,)

            classfier.trainable = False
            sum_loss, rec_loss, _, att_loss, rec_acc, _, att_acc = model.train_on_batch(x,y,)

            all_sum_loss.append(sum_loss)
            all_rec_loss.append(rec_loss)
            all_rec_acc.append(rec_acc)
            all_att_loss.append(att_loss)
            all_att_acc.append(att_acc)
            all_dis_loss.append(dis_loss)
            all_dis_acc.append(dis_acc)

            if count%200 == 0:
                train_ratio = AVG([count/len(train_generator)*100])
                print(train_ratio,'%',count)
                print(count,'Rec Loss: ',AVG(all_rec_loss),'  Dis Loss: ',AVG(all_dis_loss),'  Att Loss: ',AVG(all_att_loss),'  Sum Loss: ',AVG(all_sum_loss))
                print(count,'Rec Acc: ',AVG(all_rec_acc),'  Dis Acc: ',AVG(all_dis_acc),'  Att Acc: ',AVG(all_att_acc))
            count += 1