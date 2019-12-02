import sklearn.svm
import datasets
import numpy as np
#load the data
x_train, y_train, x_test, y_test = datasets.gaussian_dataset(n_train=800, n_test=800)

# choices=['linear','poly', 'rbf', 'sigmoid']
#Linear kernel
model = sklearn.svm.SVC(C=1, kernel='linear')
train=model.fit(x_train,y_train)
test=model.predict(x_test)
print("accuracy for gaussian linear kernel: "  + str(np.mean(test == y_test)))
######################################################
x_train, y_train, x_test, y_test = datasets.moon_dataset(n_train=800, n_test=800)
#Linear kernel
model = sklearn.svm.SVC(C=10,gamma=0.001, kernel='linear')
train=model.fit(x_train,y_train)
test=model.predict(x_test)
print("accuracy for half_moon linear kernel: "  + str(np.mean(test == y_test)))

#radial basis kernel
model = sklearn.svm.SVC(C=10, kernel='rbf')
train=model.fit(x_train,y_train)
test=model.predict(x_test)
print("accuracy for rbf: "  + str(np.mean(test == y_test)))
#hyperbolic tangent kernel
model = sklearn.svm.SVC(C=100, gamma=0.01, kernel='sigmoid')
train=model.fit(x_train,y_train)
test=model.predict(x_test)
print("accuracy for sigmoid: "  + str(np.mean(test == y_test)))
#polynomial kernel
model = sklearn.svm.SVC(C=100,gamma=0.1, degree=3, kernel='poly')
train=model.fit(x_train,y_train)
test=model.predict(x_test)
print("accuracy for poly: "  + str(np.mean(test == y_test)))

