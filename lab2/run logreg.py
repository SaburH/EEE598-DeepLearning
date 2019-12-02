from logistic_regression import LogisticRegression
import numpy as np
import datasets
import matplotlib.pyplot as plt

model=LogisticRegression()
# load data
x_train, y_train, x_test, y_test = datasets.gaussian_dataset(n_train=800, n_test=800)

# loss=np.zeros(3)
lr=[0.1, 0.01, 50]
model.l2_reg=0
model.n_epochs=100
model.lr=lr[0]
loss1=model.fit(x_train, y_train)
# loss[0]= model.loss((x_train,y_train))


model.lr=lr[1]
loss2= model.fit(x_train,y_train)

model.lr=lr[2]
loss3=model.fit(x_train,y_train)

# j=0
# accuracy =np.zeros(len(r))
# for i in range(1,51,5):
#     model = KNN(k=i)
#     model.fit(x_train, y_train)
#
#     y_pred = model.predict(x_test)
#     accuracy[j]= np.mean(y_pred == y_test)
#     print("knn="+ str(i) + "  accuracy: "  + str(np.mean(y_pred == y_test)))
#     j+=1

plt.plot(range(0,100), loss1, 'r*-', label='L1 lr=0.1')
plt.plot(range(0,100), loss2, 'y*-', label='L2 lr=0.01')
plt.plot(range(0,100), loss3, 'b*-', label='L3 lr=50')

plt.legend()
plt.show()