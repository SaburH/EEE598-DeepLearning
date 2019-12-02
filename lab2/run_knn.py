from knn import KNN
import numpy as np
import datasets
import matplotlib.pyplot as plt


# load data
x_train, y_train, x_test, y_test = datasets.gaussian_dataset(n_train=800, n_test=800)

# model = KNN(k=3)
# model.fit(x_train, y_train)
#
# y_pred = model.predict(x_test)
# print("knn accuracy: " + str(np.mean(y_pred == y_test)))
j=0
r=range(1,51,5)
accuracy =np.zeros(len(r))
for i in range(1,51,5):
    model = KNN(k=i)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy[j]= np.mean(y_pred == y_test)
    print("knn="+ str(i) + "  accuracy: "  + str(np.mean(y_pred == y_test)))
    j+=1

plt.plot(r, accuracy, 'r*', label='MSE train')
# plt.plot(range(1,51), MSETest, 'y*', label='MSE Test')
plt.legend()
plt.show()
