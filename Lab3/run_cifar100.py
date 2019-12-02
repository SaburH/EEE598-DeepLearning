

"""
=> Your Name: ِAbdulhakim Sabur

1. In this script, You need to implement the simple neural network using code presented in Section 4.1.
2. Using the network above, plot the average training loss vs epoch for learning rates 0.1, 0.01, 0.3 for 20 epochs.
3. Run the same network with learning rate 10 and observe the result.
4. Report the test accuracy with learning rates 0.1, 0.01, 0.3 and 10 for 20 epochs.

=> After running this script, describe below what you observe when using learning rate 10:
When using learning rate of 10, the learned parameters exploed, which will results in some zero values that can't be entered into the log() and will
result in a NaN output. Thus, the accuracy for this learning rate is actually the probaiblity of selecting the correct class from the 5 classes which 
is 1/5 (20%).

"""
import numpy as np
import matplotlib.pyplot as plt
from layers import (FullLayer,ReluLayer,SoftMaxLayer,CrossEntropyLayer,Sequential)
from layers.dataset import cifar100
model1 =Sequential(layers=(FullLayer(32*32*3,500),
                          ReluLayer(),
                          FullLayer (500 , 5) ,
                          SoftMaxLayer() ),
                  loss=CrossEntropyLayer ( ))

(x_train, y_train), (x_test, y_test),selected_cats =  cifar100(1213187844)
#Fit the model

loss1= model1.fit(x_train,y_train,20,0.01,128)
y_predict=model1.predict(x_test)
compare=y_predic t== y_test
accuracy1= (np.sum(compare))*1/5


model2 =Sequential(layers=(FullLayer(32*32*3,500),
                          ReluLayer(),
                          FullLayer (500 , 5) ,
                          SoftMaxLayer() ),
                  loss=CrossEntropyLayer ( ))
loss2= model2.fit(x_train,y_train,20,0.1,128)
y_predict=model2.predict(x_test)
compare=y_predict==y_test
accuracy2=(np.sum(compare))*1/5


model3 =Sequential(layers=(FullLayer(32*32*3,500),
                          ReluLayer(),
                          FullLayer (500 , 5) ,
                          SoftMaxLayer() ),
                  loss=CrossEntropyLayer ( ))
loss3= model3.fit(x_train,y_train,20,0.3,128)
y_predict=model3.predict(x_test)
compare=y_predict==y_test
accuracy3=(np.sum(compare))*1/5

model4 =Sequential(layers=(FullLayer(32*32*3,500),
                          ReluLayer(),
                          FullLayer (500 , 5) ,
                          SoftMaxLayer() ),
                  loss=CrossEntropyLayer ( ))
loss4= model4.fit(x_train,y_train,20,10 ,128)
y_predict=model4.predict(x_test)
compare=y_predict==y_test
accuracy4=(np.sum(compare))*1/5

print("Accuracy 1: " + str(accuracy1)+"%")
print("Accuracy 2: " + str(accuracy2)+"%")
print("Accuracy 3: " + str(accuracy3)+"%")
print("Accuracy 4: " + str(accuracy4)+"%")

plt.plot(range(0,20), loss1, 'r*-', label='L1 lr=0.01')
plt.plot(range(0,20), loss2, 'y*-', label='L2 lr=0.1')
plt.plot(range(0,20), loss3, 'b*-', label='L3 lr=0.3')


plt.legend()
plt.show()

