import pandas as pd 
import keras
import numpy as np
import csv
import sys

def start_predict(testpath = './test.csv', anspath = './ans.csv'):
	df_test = pd.read_csv(testpath)
	model = keras.models.load_model('1543385951.h5')
	x_test = []
	for i in range(np.shape(df_test)[0]):
		tem = np.array(df_test['feature'][i].split())
		tem = tem.astype(np.float)
		tem = np.reshape(tem,(48,48,1))
		x_test.append(tem/255)

	x_test = np.array(x_test)

	predict = model.predict_classes(x_test)

	with open(anspath, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['id', 'label'])
		
		for l in range(len(predict)):
			ids = str(l)
			writer.writerow([ids, predict[l]])
		
if __name__ == '__main__':
    if len(sys.argv) != 1:
        
        start_predict(sys.argv[1],sys.argv[2])   