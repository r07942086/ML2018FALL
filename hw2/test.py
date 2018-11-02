import numpy as np
import csv
import sys



def px(x,u,cov):
    d = len(x)
    inv = np.linalg.inv(cov)
    a1 = 1/np.power(2*np.pi,d/2)*(1/np.power(np.linalg.det(cov),0.5))
    a2 = np.exp(-0.5* np.dot(np.dot((x-u), inv) ,(x-u)))

    return a1*a2

def start_predict(trainxpath='./train_x.csv', trainypath = './train_y.csv',testpath = './test_x.csv', anspath = './ans.csv'):

    ally = []
    with open(trainypath, 'r') as f:
        lines = f.readlines()
        for line_i in range(1,len(lines)):
            things = lines[line_i].replace('\n', "").split(',')
    
            ally.append(float(things[0]))
    
        
        
    with open(trainxpath, 'r') as f:
        lines = f.readlines()
    
        allx = []
        alldata1=[[]for i in range(29)]
        alldata2=[[]for i in range(29)]
        
        for line_i in range(1,len(lines)):
            row = []
            things = lines[line_i].replace('\n', "").split(',')
            for j in range(23):
                row.append(float(things[j]))
            for k in range(6):
                row.append((float(things[11+k])-float(things[17+k]))/ float(things[0]))
            
    
            row = np.array(row)
            
            allx.append(np.array(row))
            in_size = len(row)
            
    
                
            for l in range(in_size):
                if ally[line_i-1]==float(1):
                    alldata1[l].append(row[l])
                else:
                    alldata2[l].append(row[l])
        
        alldata1 = np.array(alldata1)
        alldata2 = np.array(alldata2)        
        class1mean = []
        class2mean = []
        
         
        for l in range(in_size):
            class1mean.append(np.mean(alldata1[l]))
            class2mean.append(np.mean(alldata2[l]))   
        
        class1mean = np.array(class1mean)
        class2mean = np.array(class2mean)
        cov1 = np.cov(alldata1)
        cov2 = np.cov(alldata2)    
        
        cov = (np.shape(alldata1)[1]/ 20000) * cov1 + (np.shape(alldata2)[1]/20000) * cov2
        
        
    with open(testpath, 'r') as f:
        lines = f.readlines()
    
        testx = []
        
        for line_i in range(1,len(lines)):
            row = []
            things = lines[line_i].replace('\n', "").split(',')
            for j in range(23):
                row.append(float(things[j]))
            for k in range(6):
                row.append((float(things[11+k])-float(things[17+k]))/ float(things[0]))
    
            
            row = np.array(row)
    
            
            testx.append(np.array(row))
                
        
    
        pr = []
        with open(anspath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id', 'Value'])
            
            for l in range(10000):
                ids = 'id_'+ str(l)

                pre1 = px(testx[l],class1mean,cov) * 4445/20000
                pre2 = px(testx[l],class2mean,cov) * 15555/20000
                pre = pre1/(pre1+pre2)
                pr.append(pre)
                if pre>0.5:
                    writer.writerow([ids, '1'])
                else:
                    writer.writerow([ids, '0'])
    
if __name__ == '__main__':
    if len(sys.argv) != 1:
        
        start_predict(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])   