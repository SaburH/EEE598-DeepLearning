import numpy as np
import scipy.signal as sig


class ConvLayer(object):
    def __init__(self, n_i, n_o, h):
        """
        Convolutional layer

        Parameters
        ----------
        n_i : integer
            The number of input channels
        n_o : integer
            The number of output channels
        h : integer
            The size of the filter
        """
        # glorot initialization
        # raise NotImplementedError

        f_in = n_i*h*h
        f_out = n_o*h*h
        self.W = np.sqrt( 1.0/(f_in+f_out) )*np.random.randn( n_o, n_i, h, h )
        self.b = np.zeros(n_o)
        self.W_grad = None
        self.b_grad = None
        self.input = None

    def forward(self, x):
        """
        Compute "forward" computation of convolutional layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of input channels x number of rows x number of columns

        Returns
        -------
        np.array
            The output of the convolutiona

        Stores
        -------
        self.x : np.array
             The input data (need to store for backwards pass)
        """
        n_b = x.shape[0]
        n_o = self.W.shape[0]
        n_r = x.shape[2]
        n_c = x.shape[3]
        # Zero-padding loop
        p = np.floor( self.W.shape[3]/2.0 ).astype(int)
        x_padded = np.pad(x, ((0,0),(0,0),(p,p),(p,p)), 'constant', constant_values=0)
        f = np.zeros( (n_b,n_o,n_r,n_c) )
        # Computing output feature maps:
        feat_ind = 0
        for sample in x_padded:
            kernel_ind = 0
            for k in range(self.W.shape[0]):# looping over the different filters
                f[feat_ind,kernel_ind,:,:] = sig.correlate(sample,self.W[k,:,:,:],mode='valid') + self.b[ kernel_ind  ]
                kernel_ind += 1
            feat_ind += 1
        self.input = x
        return f

    def backward(self, y_grad):
        """
        Compute "backward" computation of convolutional layer

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input

        Stores
        -------
        self.b_grad : np.array
             The gradient with respect to b (same dimensions as self.b)
        self.w_grad : np.array
             The gradient with respect to W (same dimensions as self.W
        """
        # Gradient of b
        self.b_grad = np.sum( y_grad,axis=(0,2,3) )

        # Gradient of input
        dLdx = np.zeros((y_grad.shape[0], self.W.shape[1], y_grad.shape[2], y_grad.shape[3]))
        p = np.floor(self.W.shape[3] / 2.0).astype(int)
        y_padded = np.pad(y_grad, ((0, 0), (0, 0), (p, p), (p, p)), 'constant', constant_values=0)
        # y_padded = y_grad
        for j in range(y_padded.shape[0]):
            for i in range(self.W.shape[1]):
                w = self.W[:,i,:,:]
                w = np.flip( np.flip(w, axis=1), axis=2)
                dLdx[j,i,:,:] = sig.correlate(y_padded[j,:,:,:], w, mode='valid')


        # Gradient of W:
        self.W_grad = np.zeros(self.W.shape)
        x_padded = np.pad(self.input, ((0, 0), (0, 0), (p, p), (p, p)), 'constant', constant_values=0)
        for j in range(y_grad.shape[1]):# looping over y_grad channels
            for i in range(x_padded.shape[1]):# looping over x_padded channels
                self.W_grad[j,i,:,:] = sig.correlate(x_padded[:,i,:,:], y_grad[:,j,:,:], mode='valid')

        return dLdx


        # raise NotImplementedError

    def update_param(self, lr):
        """
        Update the parameters with learning rate lr

        Parameters
        ----------
        lr : floating point
            The learning rate

        Stores
        -------
        self.W : np.array
             The updated value for self.W
        self.b : np.array
             The updated value for self.b
        """
        # raise NotImplementedError
        self.W = self.W - (lr*self.W_grad)
        self.b = self.b - (lr*self.b_grad)