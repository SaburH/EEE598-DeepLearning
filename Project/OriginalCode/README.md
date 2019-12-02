-------------------------------------
  Requirements
-------------------------------------
Python 3.7

Pytorch 1.0.1 (with CUDA support)

Torchvision

CUDA (version depends on the installed GPU. In our implementation: CUDA 10)

numpy 1.15.4

skimage

shutil

---------------------
 Referenced paper
 --------------------

 Original paper title: Deep Residual Learning for Image Recognition.
		   Citation: He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.	"Deep residual learning for image recognition." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778. 2016."	

--------------------
 Original Code
 -------------------

 Original Source Code can be found in the following url: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

--------------------
 Dataset Link
 -------------------

 Dataset can be downloaded using this url: https://drive.google.com/file/d/1_JPJ9SpqTJBjnvSF6gEWDbFp4QZFrUnq/view?usp=sharing 
 Note: please download the validation images (all tasks) size: 6.3GB"

--------------------
 Folder structure
 -------------------

 Listing of files in this folder:

1) classsification_test.py file: this is the file where we load the data into pytorch data loader and do the model verification and testing.
	
2) buildNetwork.py file: this is the original source code of ResNet where it has multiple models. We commented out the models we don't use.

3) valprep.sh file: IMPORTANT: please put this file inside the folder where you unzipped all the downloaded images and run it. This shell code will restructure the validation data so that  the validation test can run. If you downloaded the dataset from the website, then this shell code must be run in the same folder where you downloaded the validation data. 

------------------------
 Running Instructions 
 -----------------------

 Steps to run the code: 
 
1) Data preparation: Please download the dataset from the above provided URL, then, unzip it in the folder OriginalCode.
	
2) Run the file classsification_test.py.  After running it, you will get the accuracy for the model which is 76.13%, the same accuracy reported in Pytorch website.




 


