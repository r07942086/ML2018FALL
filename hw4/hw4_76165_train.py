import sys
import pandas as pd 
import numpy as np
import re

def start_predict(train_x_path='./train_x.csv', train_y_path='./train_y.csv', test_x_path='./test_x.csv', dict_path = './dict.txt.big'):

  import jieba
  jieba.load_userdict(dict_path) 

  def Q2B(uchar):
    inside_code=ord(uchar)
    if inside_code==0x3000:
      inside_code=0x0020
    else:
      inside_code-=0xfee0
    if inside_code<0x0020 or inside_code>0x7e:
      return uchar
    return chr(inside_code)

  def Q2BS(instr):
    r = ''
    for s in instr:
      r+=Q2B(s)
    return r

  DIC_DIM = 256
  INPUT_LEN = 128
  dic = []

  jx = []
  y = []

  with open(train_y_path, 'r',encoding = 'utf_8') as file:
    lines = file.readlines()
    for i in range(1,len(lines)):
      tem = lines[i][:-1].split(',')
      if tem[1] == '1':
        y.append(np.array([0,1]))
      else:
        y.append(np.array([1,0]))
  y = np.array(y)
  with open(train_x_path, 'r',encoding = 'utf_8') as file:
    lines = file.readlines()
    for i in range(1,len(lines)):
      lines[i] = lines[i][:-1]
      teml = lines[i].split(',')
      jtem = jieba.lcut(Q2BS(teml[1]))
      nj = []
      before = ''
      for item in jtem:
        search_floor = re.search('(b\d+)|(B\d+)', item)
        search_number =  re.search('(\d+)',item)
        if item == before:
          continue  
        elif item ==' ' or item =='\u3000' or item == ',' or item == '，'or item == '。':
          nj.append(' ')
        elif type(search_floor)!=type(None)  and search_floor.group(0)==item:
          nj.append('Dcard用戶')
#        elif type(search_number)!=type(None) and search_number.group(0)==item and item!='9.2':
#          nj.append('某種數字ㄜ')
        else:
          nj.append(item)
        before = item
      jx.append(nj)
      dic.append(nj)


  with open(test_x_path, 'r',encoding = 'utf_8') as file:
    lines = file.readlines()
    for i in range(1,len(lines)):
      lines[i] = lines[i][:-1]
      teml = lines[i].split(',')
      jtem = jieba.lcut(Q2BS(teml[1]))
      nj = []
      before = ''
      for item in jtem:
        search_floor = re.search('(b\d+)|(B\d+)', item)
        search_number =  re.search('(\d+)',item)
        if item == before:
          continue
        elif item ==' ' or item =='\u3000' or item == ',' or item == '，'or item == '。':
          nj.append(' ')
        elif type(search_floor)!=type(None)  and search_floor.group(0)==item:
          nj.append('Dcard用戶')
#        elif type(search_number)!=type(None) and search_number.group(0)==item and item!='9.2':
#          nj.append('某種數字ㄜ')
        else:
          nj.append(item)
        before = item
      dic.append(nj)

  from gensim.models import Word2Vec

  wv_model = Word2Vec(dic, size=DIC_DIM , window=6, min_count=1, workers=5, iter = 25)
  wv_model.save("word2vec_76165.model")

  for i in range(len(jx)):
    if len(jx[i]) > INPUT_LEN:
      jx[i] = jx[i][:INPUT_LEN]
    else:
      add = INPUT_LEN - len(jx[i])
      jx[i] = jx[i] + [' ' for num in range(add)]

  wv_x = []
  for i in range(len(jx)):
    tem = []
    for word in jx[i]:
      tem.append(wv_model.wv[word])
    wv_x.append(np.array(tem))
  wv_x = np.array(wv_x)

  import keras
  from keras import layers

  model = keras.models.Sequential()
  model.add(layers.LSTM(256, input_shape=(INPUT_LEN,DIC_DIM),dropout=0.5 , recurrent_dropout=0.5, go_backwards=True))
  model.add(layers.Dense(512,activation='relu'))   
  model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.25))
  model.add(layers.Dense(512,activation='relu'))   
  model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.25))
  model.add(layers.Dense(2, activation='softmax'))
  optimizer = keras.optimizers.Adam(lr=0.0015, decay=1e-6, clipvalue=0.5)
  model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

  model.summary()
  from keras.callbacks import CSVLogger
  from keras.callbacks import ModelCheckpoint
  from keras.callbacks import EarlyStopping



  checkpoint = ModelCheckpoint(filepath='1545025935.h5',
                             verbose=1,
                             save_best_only=True,
                             monitor='val_acc',
                             mode='max')
  earlystopping = EarlyStopping(monitor='val_acc', 
                              patience=10, 
                              verbose=1, 
                              mode='max')

  history = model.fit(wv_x, y,
              batch_size=512,
              epochs=100,
              validation_split = 0.02,
              callbacks=[checkpoint, earlystopping])
                  
if __name__ == '__main__':
  if len(sys.argv) != 1:
    start_predict(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])   