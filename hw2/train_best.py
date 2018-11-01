import numpy as np
from collections import OrderedDict
import csv
import matplotlib.pyplot as plt

class Adam:

    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx

class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b
        
        self.x = None
        self.original_x_shape = None

        self.dW = None
        self.db = None

    def forward(self, x):
        
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)
        return dx

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None 
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx
    
class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, output_size)
        self.params['b1'] = np.zeros(output_size)

        
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Sigmoid'] = Sigmoid()
        
        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db

        return grads

ally = []    
with open('train_y.csv', 'r') as f:
    lines = f.readlines()
    for line_i in range(1,len(lines)):
        things = lines[line_i].replace('\n', "").split(',')
        if float(things[0]) == 0:
            tem = np.array([0,1])
            ally.append(tem)
        else:
            tem = np.array([1,0])
            ally.append(tem)
    
    
with open('train_x.csv', 'r') as f:
    lines = f.readlines()

    allx = []
    alldata=[[]for i in range(30)]
    
    for line_i in range(1,len(lines)):
        row = []
        things = lines[line_i].replace('\n', "").split(',')
        for j in range(23):
            row.append(float(things[j]))
        for k in range(6):
            row.append((float(things[11+k])-float(things[17+k]))/ float(things[0]))
        row.append(float(1))
        row = np.array(row)
        
        allx.append(np.array(row))
    
        for l in range(30):
            alldata[l].append(row[l])
            
    ms = []
    ss = []
    for l in range(30):
        ms.append(np.mean(alldata[l]))
        ss.append(np.std(alldata[l]))
    
    for i in range(len(allx)):
        for j in range(len(allx[i])-1):
            allx[i][j] = (allx[i][j] - ms[j]) / ss[j]
    np.save('mean',ms)
    np.save('std',ss)
    
    network = TwoLayerNet(input_size=30, hidden_size=2, output_size=2)
    allx = np.array(allx)
    ally = np.array(ally)
    
    iters_num = 70000
    train_size = allx.shape[0]
    batch_size = 200
    learning_rate = 0.1

    iter_per_epoch = max(train_size / batch_size, 1)
    
    train_loss_list = []
    train_acc_list = []
    optimizer = Adam()
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = allx[batch_mask]
        t_batch = ally[batch_mask]
        

        grad = network.gradient(x_batch, t_batch)
        

        for key in ('W1', 'b1'):
            optimizer.update(network.params, grad)
        
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(allx, ally)
            train_acc_list.append(train_acc)   
    
    plot_x = [z for z in range(len(train_loss_list))]
    plt.plot(plot_x,train_loss_list)
    
    np.save("w_"+str(loss),network.params['W1'])
    np.save("b_"+str(loss),network.params['b1'])
            
# =============================================================================
# with open('test_x.csv', 'r') as f:
#     lines = f.readlines()
# 
#     testx = []
#     
#     for line_i in range(1,len(lines)):
#         row = []
#         things = lines[line_i].replace('\n', "").split(',')
#         for j in range(23):
#             row.append(float(things[j]))
#         for k in range(6):
#             row.append((float(things[11+k])-float(things[17+k]))/ float(things[0]))
#         row.append(float(1))
#         
#         row = np.array(row)
#         row.reshape([30,])
#         
#         testx.append(np.array(row))
#             
#     
#     for i in range(len(testx)):
#         for j in range(len(testx[i])-1):
#             testx[i][j] = (testx[i][j] - ms[j]) / ss[j]
# 
#     pr = []
#     with open('ans'+str(loss)+'.csv', 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['id', 'Value'])
#         
#         for l in range(10000):
#             ids = 'id_'+ str(l)
#             tem = testx[l]
#             pre = network.predict(np.reshape(testx[l],(1,30)))
#             pr.append(pre)
#             if pre[0][0]>0.5:
#                 writer.writerow([ids, '1'])
#             else:
#                 writer.writerow([ids, '0'])
# =============================================================================
