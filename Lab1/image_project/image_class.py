import numpy as np
import imageio
import scipy.signal
import scipy.misc
import matplotlib.pyplot as plt



class ImageClass(object):
    def __init__(self, path):
        """ Loads the image specified by path """
        self.im = imageio.imread(path).astype('uint8')
        # self.im = scipy.misc.imread(path)

    def show(self):
        """ Shows the image using Matplotlib's pyplot """
        if len(self.im.shape) == 3:
            plt.imshow(self.im)
        else:
            plt.imshow(self.im, cmap="gray")
        plt.show()

    def save(self, path):
        """ Saves the image to path """
        imageio.imsave(path, self.im)

    def crop(self, r, c, h, w):
        """
        Return a cropped version of the input with size w x h
        Parameters
        ----------
        r : int
            The row index to start the crop.
        c : int
            The column index to start the crop.
        h : int
            The height of the crop.
        w : int
            The width of the crop.
        """
        self.im=self.im[r:r+h, c:c+w,:]

        # raise NotImplementedError

    def flip_horizontal(self):
        """ Flip the image horizontally """
        # raise NotImplementedError
        self.im=np.fliplr(self.im)
    def transpose(self):
        """ Transpose """
        # raise NotImplementedError

        self.im=np.transpose(self.im,(1,0,2))

    def reverse_channels(self):
        """ Reverse the RGB channels to BGR """
        # raise NotImplementedError
        self.im=self.im[...,::-1]
    def split_channels(self):
        """
        Return a list of Numpy arrays corresponding to the three channels

        E.g. return (red, green, blue)
        """
        # raise NotImplementedError
        # plt.imshow(b, cmap="gray")
        r,g,b = self.im[:, :, 0], self.im[:, :, 1], self.im[:, :, 2]
        return r,g,b
    def to_grayscale(self):
        """ Convert the image to grayscale """
        # raise NotImplementedError
        r,g,b = self.split_channels()
        gray = (r * 0.30) + (g * 0.59) + (b * 0.11)
        gray = gray.astype("uint8")
        # plt.imshow(gray, cmap=plt.get_cmap('gray'))
        self.im=gray
        return gray

    def blur(self):
        """ Blur the image """
        # raise NotImplementedError
        r, g, b  = self.split_channels()
        ones = np.ones([7, 7])/49
        r=r.astype("float")
        g=g.astype("float")
        b=b.astype("float")

        convG = scipy.signal.convolve2d(g, ones, mode='valid')
        convR = scipy.signal.convolve2d(r, ones, mode='valid')
        convB = scipy.signal.convolve2d(b, ones, mode='valid')
        blur = np.stack((convR.astype("uint8"), convG.astype("uint8"), convB.astype("uint8")), axis=2)
        # blur.astype("uint8")
        self.im=blur
    def plot_histogram(self, show=True):
        """
        Plot and return the 1D histogram of the grayscale version of the image
        """
        # raise NotImplementedError
        self.to_grayscale()
        im=self.im.astype('float')
        histo, edg = np.histogram(im, bins=256, range=(0,255))
        plt.hist(histo)
        plt.title("Histogram with 256 bins")
        plt.show()
        return histo

    def compute_dft(self):
        """ Return the dft of the image """
        # raise NotImplementedError
        w=self._DFT_matrix()
        gray=self.to_grayscale()
        # p = np.dot(np.dot(w, gray), w)
        dft=np.fft.fft2(gray,norm="ortho")
        return dft
        # self.plot_save_dft(Y)

    def _DFT_matrix(self):
        """ Returns the dft matrix for the current image size """
        N = self.im.shape[0]
        i, j = np.meshgrid(np.arange(self.im.shape[0]),
                           np.arange(self.im.shape[1]))
        omega = np.exp(- 2 * np.pi * 1J / N)
        W = np.power(omega, i * j) / np.sqrt(N)
        return W

    def plot_save_dft(self, dft, fname="dft_plot.png"):
        """
        Plot and save the 2D dft of the image

        Parameters
        ----------
        dft   : complex
                2D array contain the dft of an image.
        fname : string
                The path and file name to save the 2D dft.
        """
        dft = np.log10(np.abs(dft) + 1)
        dft = np.roll(dft, dft.shape[0]/2, axis=0)
        dft = np.roll(dft, dft.shape[1]/2, axis=1)
        dft = dft - dft.min()
        dft = dft / dft.max()
        plt.imshow(dft, cmap='gray')
        scipy.misc.imsave(fname, dft)
        plt.show()

ImageClass('sparky.png')
