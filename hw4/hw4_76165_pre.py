import pandas as pd 
import numpy as np
import re
import sys

def start_predict(test_x_path='./test_x.csv', dict_path = './dict.txt.big', anspath = './ans.csv'):
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

  DIC_DIM  = 256
  INPUT_LEN = 128


  test_jx = []

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
  #      elif type(search_number)!=type(None) and search_number.group(0)==item and item!='9.2':
  #        nj.append('某種數字ㄜ')
        else:
          nj.append(item)
        before = item
      test_jx.append(nj)


  from gensim.models import Word2Vec

  #wv_model = Word2Vec(dic, size=200, window=5, min_count=1, workers=4, iter = 10)
  wv_model = Word2Vec.load("word2vec_76165.model")

  for i in range(len(test_jx )):
    if len(test_jx[i]) > INPUT_LEN:
      test_jx[i] = test_jx[i][:INPUT_LEN]
    else:
      add = INPUT_LEN - len(test_jx[i])
      test_jx[i] = test_jx[i] + [' ' for num in range(add)]

  wv_test = []
  for i in range(len(test_jx)):
    tem = []
    for word in test_jx[i]:
      tem.append(wv_model.wv[word])
    wv_test.append(np.array(tem))
  wv_test = np.array(wv_test)


  import keras
  from keras.models import load_model

  model = load_model('1545025935.h5')
  predict = model.predict_classes(wv_test)

  import csv

  with open(anspath, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'label'])
    for l in range(len(predict)):
      ids = str(l)
      writer.writerow([ids, str(predict[l])])

if __name__ == '__main__':
  if len(sys.argv) != 1:
    start_predict(sys.argv[1],sys.argv[2], sys.argv[3])   