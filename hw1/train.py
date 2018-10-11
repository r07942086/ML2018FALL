import csv
import numpy as np
import matplotlib.pyplot as plt

w = np.zeros(162)
b = np.zeros(1)
losses = []
with open('train.csv', 'r') as f:
    lines = f.readlines()

    alldata = [[] for i in range(18)]
    for line_i in range(1,len(lines)):

        things = lines[line_i].replace('\n', "").split(',')
        for thing in things[3:]:
            if (line_i-1)%18 ==10 :
                if thing == 'NR':
                    alldata[(line_i-1) % 18].append(float(0))
                else:
                    alldata[(line_i-1)%18].append(float(thing))
            else:
                alldata[(line_i-1)%18].append(float(thing))
    data_len = len(alldata[0])

    for i in range(18):

        alldata[i] = np.array(alldata[i])
        h_bias = np.percentile(alldata[i],99.7)
        for n in range(len(alldata[i])):
            if alldata[i][n] > h_bias:
                #alldata[i][n] = np.percentile(alldata[i],50)
                alldata[i][n] = alldata[i][n-1]



    all_x = []
    all_y = []
    data = 0
    couu = 0
    while data < data_len -10 : 
        
    #for data in range(data_len-10):   
        rowx = []

        for i in range(18):
            for j in range(9):
                rowx.append(alldata[i][data+j])
            if i == 9:
                all_y.append(alldata[i][data+9])
        
        all_x.append(rowx)
        data = data + 1
        couu = couu + 1
        if couu % 471 == 0:
            data = data + 9
            
            couu = 0

    lr = 0.000001



    losses = []
    loss = 0
    gradient_w = np.zeros([162,])
    gradient_b = np.zeros([1,])
    number = 0
    landa = 0
    w_2 = np.zeros([162,])
    
    #for k in range(len(all_x)):
    for k in np.random.randint(int(len(all_x)), size=1200000):

        predict = np.dot(w,all_x[k])+b


        loss +=(all_y[k] -  predict) ** 2

        gradient_w += -2 *   (all_y[k] -  predict)  * all_x[k]
        gradient_b += -2 *   (all_y[k] -  predict)
        number = number + 1

        if number % 50 == 0:
            gradient_w = gradient_w / number
            gradient_b = gradient_b / number
            
            

            w =  w * (1 - lr * landa/number) - ( gradient_w * lr) #regulization
            #w =  w - ( gradient_w * lr)
            b = b * (1 - lr * landa/number) -  gradient_b * lr
            losses.append((loss/number))
            loss = 0
            gradient_w = np.zeros([162,])
            gradient_b = np.zeros(1)
            number = 0
        

    plot_x = [z for z in range(len(losses))]
    
    plt.plot(plot_x, losses)
    plt.show()

    loss = 0
    avg = 0
    for _ in range(5):
        cou = 0
        for k in  np.random.randint(int(len(all_x)), size=500):
    
            predict = np.dot(w,all_x[k])+b
            loss += (all_y[k] - predict) ** 2
            cou = cou +1
            
        avg += loss/cou
        
    avg = avg/5
    print(avg)
    np.save('weight_'+str(avg),w)
        

with open('test.csv', 'r') as f:
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
    
    with open('ans_'+str(avg)+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'value'])
        
        for l in range(260):
            ids = 'id_'+ str(l)
            pre = np.dot(w,all_test[l])+b   
            writer.writerow([ids, float(pre)])