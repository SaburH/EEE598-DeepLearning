Revised Code README file.
Selected topic: Content-based image retrieval.

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

----------------------------------------------
  Referenced paper and link to existing code
----------------------------------------------      

Paper:
 
 R. Arandjelović, P. Gronat, A. Torii, T. Pajdla and J. Sivic, "NetVLAD: CNN Architecture for Weakly Supervised Place Recognition," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 40, no. 6, pp. 1437-1451, 1 June 2018.

Links to existing code: 

Original code: https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py

-----------------------------------------
  Links to existing dataset
----------------------------------------- 

1) Training set for NetVLAD: https://drive.google.com/file/d/18Lje0LhZ-6nI8XHA_ex9Uq6iS7MUp40-/view?usp=sharing


-------------------------------------------------------
  Listing of sub-folders and files under this folder:
-------------------------------------------------------

1) oxbuild_images folder: It contains two sub-folders, namely query and database. In the query sub-folder, we manually divided all query images from oxford dataset into sub-sub-folders according to their respective classes. The database sub-folder includes all other images not included in the query folder.
2) oxford_groundTruth folder: This folder contains lists of ground truth, good and ok matches, to every query image. 
3) paris_120310 folder: This folder contains lists of ground truth files for good, ok and junk files as marked by the dataset authors. Junk files will be removed by file_splitting.py script. 
4) buildNetVLAD.py file: builds NetVLAD network. It is our implementation inspired by the code in file OriginalCode_NetVLAD.py.
5) buildNetwrok.py file: builds the base network for NetVLAD, which is ResNet50. 
6) feedNetVLAD.py  file: it includes two data feeding classes: the first feeds training data for training NetVLAD & the other feeds database data for testing NetVLAD.
7) files_splitting.py file: (DO NOT RUN) We used this file to organize the Paris dataset and make our own training set.
8) NetVLAD_train.py: The main file that trains NetVLAD using the restructured Paris dataset.
9) NetVLAD_test.py file:  The main file that tests NetVLAD performance on the restructured oxford dataset.
10) not_removed.txt file: this file assists files_splitting.py code in cleaning the paris dataset from the marked junk files.
11) OriginalCode_NetVLAD.py file:(NOT USED)   This is the implementation that guided us through the implemntatin of buildNetVLAD.py. 
12) precision_top_1.npy - precision_top_20.npy files:  These are numpy files that contain the precision values  at the end of NetVLAD test. 
13) recall_top_1.npy - recall_top_20.npy files: These are numpy files that contain the recall values at the end of NetVLAD test.  
14) trainedVLAD_VLADLayer_epoch5 file: This file contains the trained NetVLAD obtained at the end of the 5th training epoch. We found that these parameters give the best performance in comparison to using 1,2,3 and 4 epochs.
15) TestResults_NetVLAD.txt file: This file has the test results of NetVLAD.
---------------------------------------------------
  Description of implemented transfer learning
  -------------------------------------------------
We choose to do finetuning of ResNet50 for the purpose of tackling image retrieval problems.
In our work, we discard the last two layers of ResNet50, namely average pooling and fully-connected layer.
We, then, use it as the base network from which local image descriptors are extracted.
Those descriptors are fed to a new layer, called Vectors of Locally Aggregated Descriptors (VLAD), that combines them into a single "Global descriptor". 
The whole architecture, called NetVLAD, is trained end-to-end to learn two thing: 1) how to extract better local descriptors,
 and 2) the parameters of the VLAD layer.
--------------------------------------------------
  Motivation behind revised implementation
--------------------------------------------------
In large image database, it is important to have fast retrieval methods. Processing a single image relatively consumes long time.
Using image global descriptors as primary key for image retrieval has proven to be fast. Thus, NetVLAD attempts to leverage the
feature-extraction ability of impressive classification networks, like ResNet50, and learn how to generate a single and highly-
discriminative global descriptor.

--------------------------------------------------
 Step-by-step running instructions
--------------------------------------------------

Dataset download:
Please download the datasets from this link, unzip the files and put them in the same folder "RevisedCode":
https://drive.google.com/drive/folders/1t61p2f3o4XyIdPLCbbdxHR4CRSMhmTYR?usp=sharing 


Testing our trained model:

Step 1: To test NetVLAD, run the file 'NetVLAD_test.py' and you will get the results for testing. 
We have provided a code that will automatically load trainedVLAD_VLADLayer_epoch5, which is the model 
we trained, and prints out the final results. NOTE: this may take around 30 mins to run on a single GPU 

Training NetVLAD:

Step 1: Download the training dataset from the following link: https://drive.google.com/file/d/18Lje0LhZ-6nI8XHA_ex9Uq6iS7MUp40-/view?usp=sharing 

Step 2: unzip the downloaded file and put the folder 'training_set' inside the RevisedCode folder.

step 3: Run NetVLAD_train.py file. 


------------------------------------------------
 Output Format:
 -----------------------------------------------
 The script in NetVLAD_test.py considers four retrieval cases, which are: 
 top-1, top-5, top-10, and top-20. For each case, it prints the following:
 
 1) Name of category
 
 2) Precisions and recalls for all queries in one category.
 
 3) Average precision and recall over all queries (computed using those values in (1)).

Note: we have provided the output results that we got in the file " TestResults_NetVLAD.txt".
 
 
