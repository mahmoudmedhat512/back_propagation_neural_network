# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:46:05 2019

@author: revan
"""
import numpy as np
import sys
class NN:
     def __init__(self,m,l,n):
         self.m = m
         self.l = l
         self.n = n
         self.hidden_weights = np.random.randn(self.m, self.l)
         self.output_weights = np.random.randn(self.l, self.n)
     def sigmoid(self,s, prime = False):
         
         if(prime):
             return s * (1 - s)
    
         return 1/(1+np.exp(-s))
     def SOP(self,x,layer_weights, bais = 0):
         return np.dot(x,layer_weights)+bais
     
     def feed_forward(self,x):
         self.hidden_layer_out = self.sigmoid(self.SOP(x, self.hidden_weights))
         output_layer_out = self.sigmoid(self.SOP(self.hidden_layer_out, self.output_weights))
         return output_layer_out
     
     
     def calc_error(self,actual , out):
         return np.sum((np.subtract(actual, out))**2)/len(out)
     # not complated yet
     def update_weights(self,x,y,out,eta):
         ''' Parameters
         ----------
         x : input data 
         y : acual output 
         error : output from forward pass
         eta : learing_rate
         Returns
         ------  
         '''
         error = y-out
         delta_out = eta*error*self.sigmoid(out)
         W2 = np.dot(self.hidden_layer_out.T,delta_out)
         W1 = np.dot(np.transpose(x),np.dot(delta_out,self.output_weights.T))
         
         #print("W2",W2.shape)
         #print("W1",W1.shape)
         self.hidden_weights+=W1
         self.output_weights+=W2

         
         
         
         
     def backPropugation(self,x,y,rate):
         out = self.feed_forward(x)
         self.update_weights(x,y,out,rate)
         return self.calc_error(y, out)
        
         
         
         
         
         

nn_obj = NN(8,10,1)


with open('train.txt') as f:
    x = list()
    y = list()
    a  = f.readline().split()
    m = int(a[0])
    l = int(a[1])
    n = int(a[2])
    f.readline()
    for line in f:
        line = line.split() # to deal with blank 
        if line:            # lines (ie skip them)
            line = [float(i) for i in line]
            x.append(line[0:m])
            y.append(line[m:])
y = y/np.amax(y, axis=0)     
'''
predicted = nn_obj.feed_forward(x[0])
print("before")
print(nn_obj.output_weights)
print("hidden >>>>",nn_obj.hidden_weights.shape)

nn_obj.update_weights(x, y, predicted,0.01)
print("after")
print(nn_obj.output_weights)
print("hidden >>>>",nn_obj.hidden_weights.shape)


mm = np.random.randn(8)
mm2 = np.random.randn(10)
for i in range (8):
    print(mm[i]*mm2)

'''


out = nn_obj.feed_forward(x)
MSE = 0

for i in range(500):
    MSE = nn_obj.backPropugation(x, y, 0.01)
    if(MSE <=0.025):
        break
   
print("MSE => ",MSE)     

with open('new_weights.txt', 'w') as f:
    for item in nn_obj.hidden_weights:
        f.write("%s\n" % item)
    for item in nn_obj.output_weights:
        f.write("%s\n" % item)
    f.close()