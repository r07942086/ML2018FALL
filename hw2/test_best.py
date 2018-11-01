import csv
import numpy as np
import sys
from collections import OrderedDict


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
        self.params['W1'] = np.load('w_0.5229725130656057.npy')
        self.params['b1'] = np.load('b_0.5229725130656057.npy')

        
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

def start_predict(filepath = './test_x.csv', anspath = './ans.csv'):
    
 
    ss = np.load('std.npy')
    ms = np.load('mean.npy')
    
    network = TwoLayerNet(input_size=30, hidden_size=2, output_size=2)
    network.params['W1'] = np.load('w_0.5229725130656057.npy')
    network.params['b1'] = np.load('b_0.5229725130656057.npy')
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
        testx = []
        
        for line_i in range(1,len(lines)):
            row = []
            things = lines[line_i].replace('\n', "").split(',')
            for j in range(23):
                row.append(float(things[j]))
            for k in range(6):
                row.append((float(things[11+k])-float(things[17+k]))/ float(things[0]))
            row.append(float(1))
            
            row = np.array(row)
            row.reshape([30,])
            
            testx.append(np.array(row))
                
        
        for i in range(len(testx)):
            for j in range(len(testx[i])-1):
                testx[i][j] = (testx[i][j] - ms[j]) / ss[j]
    
        pr = []
        with open(anspath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id', 'Value'])
            
            for l in range(10000):
                ids = 'id_'+ str(l)

                pre = network.predict(np.reshape(testx[l],(1,30)))
                pr.append(pre)
                if pre[0][0]>0.5:
                    writer.writerow([ids, '1'])
                else:
                    writer.writerow([ids, '0'])

        
    

if __name__ == '__main__':
    if len(sys.argv) != 1:
        if 'csv' in sys.argv[3] and 'csv' in sys.argv[4]:
            start_predict(sys.argv[3],sys.argv[4])