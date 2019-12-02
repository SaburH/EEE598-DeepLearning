import numpy as np
import matplotlib.pyplot as plt
from layers import (FullLayer,ReluLayer,SoftMaxLayer,CrossEntropyLayer,Sequential)
from layers.dataset import cifar100

model1 =Sequential(layers=(FullLayer(32*32*3,2000),
                          ReluLayer(),
                           FullLayer(2000,5000),
                           ReluLayer(),
                           FullLayer(5000,1000),
                           ReluLayer(),
                          FullLayer (1000 , 5) ,
                          SoftMaxLayer() ),
                  loss=CrossEntropyLayer ( ))

(x_train, y_train), (x_test, y_test),selected_cats =cifar100(1213187844)
#Fit the model
loss1= model1.fit(x_train,y_train,20,0.01,128)
y_predict=model1.predict(x_test)
compare= y_predict == y_test
accuracy1=(np.sum(compare))*1/5
#
#
#
model2 =Sequential(layers=(FullLayer(32*32*3,2000),
                          ReluLayer(),
                           FullLayer(2000,5000),
                           ReluLayer(),
                           FullLayer(5000,1000),
                           ReluLayer(),
                          FullLayer (1000 , 5) ,
                          SoftMaxLayer() ),
                  loss=CrossEntropyLayer ( ))
loss2= model2.fit(x_train,y_train,20,0.1,128)
y_predict=model2.predict(x_test)
compare=y_predict==y_test
accuracy2=(np.sum(compare))*1/5

model3 =Sequential(layers=(FullLayer(32*32*3,2000),
                          ReluLayer(),
                           FullLayer(2000,5000),
                           ReluLayer(),
                           FullLayer(5000,1000),
                           ReluLayer(),
                          FullLayer (1000 , 5) ,
                          SoftMaxLayer() ),
                  loss=CrossEntropyLayer ( ))
loss3= model3.fit(x_train,y_train,20,0.3,128)
y_predict=model3.predict(x_test)
compare=y_predict==y_test
accuracy3=(np.sum(compare))*1/5

print("Accuracy 1: "+str(accuracy1)+"%")
print("Accuracy 2: "+str(accuracy2)+"%")
print("Accuracy 3: "+str(accuracy3)+"%")
