import numpy as np
import random

class LogisticRegression(object):
    def __init__(self, n_epochs=10, lr=0.1, l2_reg=0):
        """
        Initialize variables
        """
        self.b = None
        self.w = None
        self.n_epochs = n_epochs
        self.lr = lr
        self.l2_reg = l2_reg

    def forward(self, x):
        """
        Compute "forward" computation of logistic regression

        This will return the squashing function:
        f(x) = 1 / (1 + exp(-(w^T x + b)))

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features

        Returns
        -------
        np.array
            A 1 dimensional vector of the logistic function
        """
        # raise NotImplementedError
        # w=np.transpose(self.w)
        # x=np.transpose(x)
        ini=(np.dot(x,self.w.T)+self.b)
        return (1/(1 + np.exp(-ini))).flatten()

    def loss(self, x, y):
        """
        Return the logistic loss
        L(x) = 1/N * (ln(1 + exp(-y * (w^Tx + b)))) + 1/2 * lambda * w^T * w

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels

        Returns
        -------
        float
            The logistic loss value
        """
        # raise NotImplementedError
        # N=x.shape[0]

        step1=np.dot(x,self.w.T) +self.b
        step1=step1.flatten()
        ini= -1* np.multiply( y , step1)
        exp=np.log(1+np.exp(ini))
        lam=0.5 * self.l2_reg * np.dot(self.w, self.w.T)
        out = np.mean(exp) +lam
        if len(out)!=1:
            out=out.flatten()
        return np.asscalar(out)





    def grad_loss_wrt_b(self, x, y):
        """
        Compute the gradient of the loss with respect to b

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels

        Returns
        -------
        float
            The gradient
        """
        # raise NotImplementedError

        sum = (np.dot(x, self.w.T) + self.b)
        sum=sum.flatten()
        mult=np.multiply(y,sum)
        exp=np.exp(mult)
        denom= (1 + exp)
        divid=np.divide(-y,denom)
        return np.mean((divid))

    def grad_loss_wrt_w(self, x, y):
        """
        Compute the gradient of the loss with respect to w

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels

        Returns
        -------
        np.array
            The gradient (should be the same size as self.w)
        """
        # raise NotImplementedError
        sum=np.zeros(x.shape)

        for i in range(x.shape[0]):
            init = (np.dot(x[i,:], self.w.T) + self.b)
            exp = np.exp(init * y[i])
            denom = (1 + exp)
            nume = -1 * y[i]* x[i,:]
            sum[i,:] = (nume/denom)
        w_grad= np.mean(sum , axis=0) + (self.l2_reg*self.w)
        if len(w_grad.shape) == 1:
            w_grad= np.reshape(w_grad, self.w.shape)
        return  w_grad
    def fit(self, x, y):
        """
        Fit the model to the data

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels
        """
        # raise NotImplementedError

        self.w = np.random.randn(1,x.shape[1])
        self.b=0
        loss=np.zeros(self.n_epochs)
        yt=y.flatten()
        for i in range (0,self.n_epochs):
            b_dirv=self.grad_loss_wrt_b(x,yt)
            w_dirv = self.grad_loss_wrt_w(x, yt)

            b=self.b - (self.lr * b_dirv)
            w= self.w - (self.lr * w_dirv)
            loss[i]=self.loss(x,yt)
            self.b ,self.w = b, w
        return loss


    def predict(self, x):
        """
        Predict the labels for test data x

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features

        Returns
        -------
        np.array
            Vector of predicted class labels for every training sample
        """
        # raise NotImplementedError

        out=np.zeros(x.shape[0])
        for i in range(x.shape[0]):

            pre= self.forward(x[i,:])
            if pre <=0.5:
                out [i]= -1
            if pre > 0.5 :
                out[i]= 1
        return out.astype('int8')
