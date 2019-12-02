
from image_class import ImageClass

"""
=> Your Name:Abdulhakim Sabur

In this script, you need to use functions in ImageClass to perform the following tasks in the order specified below:
1. Load an input image
2. Compute the 2D DFT of the input image
3. Generate a plot of the DFT magnitude corresponding to the input image, and save the plot in a file called orig_img_dft.png
4. Blur the input image using a 7x7 averaging filter
5. Compute the 2D DFT of the blurred image.
6. Generate a plot of the DFT magnitude corresponding to the blurred image, and save the plot in a file called blur_img_dft.png

=> After running this script, compare the two generated DFT plots and provide your observations here:
It is noticed that bluring the image attenuates the elements in the DFT plots. The DFT transform the image into
its frequencey domain. The original image has higher frequency components as it is shown in the generated figure
On the other hand, the blured image has more zero elements in the center (which is the representation of
DFT coefficient), and element that are far from the center have higher frequency component. 

"""
image = ImageClass('sparky.png')

# image=imageio.imread("sparky.png").astype('uint8')
# imageDFT=ImageClass.compute_dft(image)
imageDFT=image.compute_dft()
image.plot_save_dft(imageDFT,"orig_img_dft.png")

blurI=ImageClass('sparky.png')
blurI.blur()
# blur=ImageClass.blur(image)
# blurDFT=ImageClass.compute_dft(blur)
blurDFT=blurI.compute_dft()
blurI.plot_save_dft(blurDFT,"blur_img_dft.png")

