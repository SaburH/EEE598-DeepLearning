class ReluLayer(object):
    def __init__(self):
        """
        Rectified Linear Unit
        """
        self.y = None
        self.mask= None


    def forward(self, x):
        """
        Implement forward pass of Relu

        y = x if x > 0
        y = 0 otherwise

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number of features

        Returns
        -------
        np.array
            The output of the layer

        Stores
        -------
        self.y : np.array
             The output data (need to store for backwards pass)
        """
        # raise NotImplementedError

        temp =x>0
        temp=temp*1.0
        y =temp * x
        self.y =y
        # self.mask=temp
        return self.y



    def backward(self, y_grad):
        """
        Implement backward pass of Relu

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input
        """
        # raise NotImplementedError
        # temp= self.mask * y_grad
        temp = self.y > 0
        output = temp*y_grad
        return  output



    def update_param(self, lr):
        pass  # no parameters to update
