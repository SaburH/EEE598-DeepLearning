from __future__ import print_function
import numpy as np


class Sequential(object):
    def __init__(self, layers, loss):
        """
        Sequential model

        Implements a sequence of layers

        Parameters
        ----------
        layers : list of layer objects
        loss : loss object
        """
        self.layers = layers
        self.loss = loss

    def forward(self, x, target=None):
        """
        Forward pass through all layers
        
        if target is not none, then also do loss layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number of features
        target : np.array
            The target data of size number of training samples x number of features (one-hot)

        Returns
        -------
        np.array
            The output of the model
        """
        # raise NotImplementedError
        # if target is not None:
        #     for i in self.layers:
        #         x=i.forward(x)


        # for i in range (len(self.layers)):
        #     # layer=self.layers[i]
        #     # x=layer.forward(x)
        #     x=self.layers[i].forward(x)
        for l in self.layers:
            x = l.forward(x)
        if target is not None:
            x= self.loss.forward(x,target)
        return x


    def backward(self):
        """
        Compute "backward" computation of fully connected layer

        Returns
        -------
        np.array
            The gradient at the input

        """
        # raise NotImplementedError
        y_grad = self.loss.backward()
        for l in reversed(self.layers):
            y_grad = l.backward(y_grad)
        return y_grad
        # for i in range(len(self.layers)):
        #     ind = -i-1
        #     y_grad= self.layers[ind].backward(y_grad)
        # return y_grad

    def update_param(self, lr):
        """
        Update the parameters with learning rate lr

        Parameters
        ----------
        lr : floating point
            The learning rate
        """
        # raise NotImplementedError
        for i in range(len(self.layers)):
            self.layers[i].update_param(lr)

    def fit(self, x, y, epochs=10, lr=0.01, batch_size=128):
        """
        Fit parameters of all layers using batches

        Parameters
        ----------
        x : numpy matrix
            Training data (number of samples x number of features)
        y : numpy matrix
            Training labels (number of samples x number of features) (one-hot)
        epochs: integer
            Number of epochs to run (1 epoch = 1 pass through entire data)
        lr: float
            Learning rate
        batch_size: integer
            Number of data samples per batch of gradient descent
        """
        n_batch = x.shape[0]/batch_size
        n_batch = np.floor(n_batch)
        out = np.zeros(epochs)
        for i in range(epochs):
            sum = 0
            print('Epoch = '+str(i))
            for j in range(int(n_batch)):
                start = int(j * batch_size)
                x_temp = x[start:start+batch_size-1,:,:,:]
                y_temp = y[start:start+batch_size-1]
                loss = self.forward(x_temp, y_temp)
                # if (j % 50) == 0:
                print('Training progress: loss is ' + str(loss))
                sum+=loss
                # print("loss= ",loss)
                self.backward()
                self.update_param(lr)
            avg = sum/float(n_batch)
            out[i]=avg
        return out



    def predict(self, x):
        """
        Return class prediction with input x

        Parameters
        ----------
        x : numpy matrix
            Testing data data (number of samples x number of features)

        Returns
        -------
        np.array
            The output of the model (integer class predictions)
        """
        # raise NotImplementedError
        output = self.forward(x)
        label=np.argmax(output,axis=1)
        return label
