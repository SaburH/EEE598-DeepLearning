from svm import SVM
import numpy as np
import datasets
import matplotlib.pyplot as plt

model=SVM()
# load data
x_train, y_train, x_test, y_test = datasets.gaussian_dataset(n_train=800, n_test=800)
# loss=np.zeros(3)
lr=[0.01, 0.1, 1]
model.l2_reg=0
model.n_epochs=100
model.lr=lr[0]
loss1=model.fit(x_train, y_train)
# loss[0]= model.loss((x_train,y_train))
model.lr=lr[1]
loss2= model.fit(x_train,y_train)

model.lr=lr[2]
loss3=model.fit(x_train,y_train)
test=model.predict(x_test)
print("  accuracy: "  + str(np.mean(test == y_test)))
plt.plot(range(0,100), loss1, 'r*-', label='L1 lr=0.01')
plt.plot(range(0,100), loss2, 'y*-', label='L2 lr=0.1')
plt.plot(range(0,100), loss3, 'b*-', label='L3 lr=1')
plt.title("SVM with lr")
plt.legend()
plt.show()