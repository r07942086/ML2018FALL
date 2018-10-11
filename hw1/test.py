import csv
import numpy as np
import sys

w = np.zeros(162)
b = np.zeros(1)

def start_predict(filepath = './test.csv', anspath = './ans.csv'):
    
    w = np.load('weight_[77.06205258].npy')
    b = np.load('b_[77.06205258].npy')
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
        alldata = [[] for i in range(18)]
        for line_i in range(0,len(lines)):
    
            things = lines[line_i].replace('\n', "").split(',')
            for thing in things[2:]:
                if (line_i)%18 ==10 :
                    if thing == 'NR':
                        alldata[(line_i) % 18].append(float(0))
                    else:
                        alldata[(line_i)%18].append(float(thing))
                else:
                    alldata[(line_i)%18].append(float(thing))
                    
        data_len = len(alldata[0])
    
        all_test = []
        for data in range(0,data_len,9):   
            rowx = []
            for i in range(18):
                for j in range(9):    
                    rowx.append(alldata[i][data+j])
            all_test.append(rowx)   

        with open(anspath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id', 'value'])
            
            for l in range(260):
                ids = 'id_'+ str(l)
                pre = np.dot(w,all_test[l])+b   
                writer.writerow([ids, float(pre)])

        
    

if __name__ == '__main__':
    if len(sys.argv) != 1:
        if 'csv' in sys.argv[1] and 'csv' in sys.argv[2]:
            start_predict(sys.argv[1],sys.argv[2])