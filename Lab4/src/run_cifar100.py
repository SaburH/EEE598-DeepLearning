import numpy as np
import matplotlib.pyplot as plt
from layers import (FullLayer,ReluLayer,SoftMaxLayer,CrossEntropyLayer,Sequential,ConvLayer,FlattenLayer,MaxPoolLayer)
from layers.dataset import cifar100


"""
=> Your Name:

In this script, you need to plot the average training loss vs epoch using a learning rate of 0.1 and a batch size of 128 for 15 epochs.

=> Final accuracy on the test set:

"""

model1 =Sequential(layers=(ConvLayer(3,16,3),
                          ReluLayer(),
                           MaxPoolLayer(),
                           ConvLayer(16,32,3),
                           ReluLayer(),
                           MaxPoolLayer(),
                           FlattenLayer(),
                          FullLayer (2048 , 3) ,
                          SoftMaxLayer() ),
                  loss=CrossEntropyLayer ( ))



(x_train, y_train), (x_test, y_test) =cifar100(1213187844)
loss1= model1.fit(x_train,y_train,15,0.1,128)


print(loss1)