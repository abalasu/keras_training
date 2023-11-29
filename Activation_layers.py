import tensorflow as tf
from keras.layers import ReLU, Softmax, Activation, Dropout
import numpy as np

# ReLU activation
a = [-1,0,-2,1,2,3,4,5,6]
a = np.array(a)
a = a.reshape(1,3,3)
x = ReLU(max_value=None, threshold=0.0)(a)
print('Input \n', a)
print('Output after Relu \n', x)

# SoftMax activation
a = [1.,2.,3.,4.,-5.]
a = np.array(a)
x = Softmax()(a)
print('Before Softmax \n', a)
print('After Softmax \n', x)
# Converting a Tensor to an np Array
x = np.array(x)
j = 0
for i in range(0,len(x)):
    j = j + x[i]
print('Total ', j)

# General Activation
a = [1.,2.,3.,4.,-5.]
a = np.array(a)
x = Activation('relu')(a)
print('Before General activation - relu \n', a)
print('After General activation - relu \n', x)

