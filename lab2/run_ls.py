"""
Run least squares with provided data
"""

import numpy as np
import matplotlib.pyplot as plt
from ls import LeastSquares
import pickle

# load data
data = pickle.load(open("ls_data.pkl", "rb"))
x_train = data['x_train']
x_test = data['x_test']
y_train = data['y_train']
y_test = data['y_test']

MSETrain=list()
MSETest=list()
ran=range(1,21)

for i in range(1,21):
    ls = LeastSquares(i)
    ls.fit(x_train, y_train)

    pred_trainX = ls.predict(x_train)
    pred_trainY= ls.predict(y_train)
    pred_test  = ls.predict(x_test)
    MSETrain.append((np.square(pred_trainX - y_train)).mean(axis=None))
    MSETest.append((np.square(pred_test - y_test)).mean(axis=None))


# plt.plot(x_test, pred_test, 'r*', label='Predicted')
# plt.plot(x_test, y_test, 'y*', label='Ground truth')
# plt.legend()
# plt.show()

plt.plot(range(1,21), MSETrain, 'r*', label='MSE train')
plt.plot(range(1,21), MSETest, 'y*', label='MSE Test')
plt.legend()
plt.show()