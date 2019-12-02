'''
The main script training and testing NetVLAD. The network is trained (End-to-End training)
on the Paris dataset found on: http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/
The dataset has been re-structured for the sake of training NetVLAD. More on that is in the
project README file. Then, it is tested on the Oxford dataset found on:
http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/
'''

import torch,os, json
import torch.cuda as cuda
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optimizer
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as trsf
from buildNetVLAD import NetVLAD
from feedNetVLAD import DataFeedTriple, FeedDB
import numpy as np


direc_to_Oxfor_images = './oxbuild_images/'
direc_to_groundTruth = './oxford_groundTruth/gt_files_170407(1)/'
list_of_classes = ['all_souls_', 'ashmolean_', 'balliol_', 'bodleian_',
                       'christ_church_', 'cornmarket_', 'hertford_', 'keble_',
                       'magdalen_', 'pitt_rivers_', 'radcliffe_camera_', ]

# --------------- A couple of helper functions ---------------#
# Compares a list of retrieved match images to the groundtruth of
# a category
def compare_images (retrieved, category, list_gt):
    match = np.zeros((len(retrieved)))
    indx=0
    for i in range(len(retrieved)):
        t=list_gt[category]
        line = retrieved[i]
        image_name = line.split(".")[0]
        check = any( x in image_name for x in t )
        if check:
            match[i] = 1
            indx +=1
    return match

# Reads ground truth images from text files and puts them into
# a dictionary of lists
def listGT():

    dict_of_lists = {}
    for i in list_of_classes:  # getting the images names of all sub-directories
        file_good = open(direc_to_groundTruth + i + str(1) + '_good.txt')
        file_ok = open(direc_to_groundTruth + i + str(1) + '_ok.txt')
        good_list_temp = file_good.readlines()
        good_list_temp += file_ok.readlines()
        good_list = []
        for s in good_list_temp:
            s = s.strip()
            s = s.split('\n')[0]
            good_list.append(s)
        dict_of_lists[i] = good_list
    return dict_of_lists

#----------------------------------------------------------------#

net = NetVLAD(dim=2048, endToEnd=True, alpha=100)# Constructs NetVLAD

########## UNCOMMENT TRAINING SECTION TO TRAIN MODEL ########
# Training NetVLAD:
# -----------------
#
# 1) prepare data loader

batch_size = 20
image_norm = trsf.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])# Normalizing image
image_resize = trsf.Resize(256)# resizing image
image_crop = trsf.CenterCrop((224,224))# center cropping image
pre_proc = trsf.Compose([
                         trsf.ToPILImage(),
                         image_resize,
                         image_crop,
                         trsf.ToTensor(),
                         image_norm
                        ])

train_set = DataFeedTriple('./training_set',transform=pre_proc)# train_set is a 3-tuple of images
data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)# gets data from training set

# 2) prepare optimizer

criterion = nn.TripletMarginLoss(margin=0.1, p=2)# loss function
opt = optimizer.SGD(filter(lambda param: param.requires_grad, net.parameters()),
                    lr = 0.001, weight_decay=0.0001)# Stochastic Gradient Descent with momentum and l2-regularization
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.5, patience=1, verbose=True)# Reduces learning rate upon loss saturation.

# 3) commence training

with cuda.device(0):
    net = net.cuda()
    net.train()
    for epoch in range(5):
        print('Epoch # '+str(epoch))
        accumLoss = 0
        for i, (q,p,n) in enumerate(data_loader):
            query = q.cuda()# Query image
            pos = p.cuda()# Positive matching image
            neg = n.cuda()# Negative image
            opt.zero_grad()
            out_q = net(query)# get descriptor of query image
            out_p = net(pos)# get descriptor of positive match
            out_n = net(neg)# get descriptor of negative image
            l = criterion(out_q, out_p, out_n)
            l.backward()
            opt.step()
            accumLoss += l.item()
            if np.mod(i,49) == 0:
                print('Loss is: '+str( accumLoss/(i+1) ))
        scheduler.step(l)
        # Save checkpoint
        net_name = 'trainedVLAD_VLADLayer_epoch'+str(epoch+1)
        torch.save(net, net_name)

# -------------------------------------------------------------------------- #
# Testing NetVLAD
# ---------------

# Loading net from file (pre-trained)
# net = torch.load('trainedVLAD_VLADLayer_epoch5')

batch_size = 5
list_gt = listGT() # get the dictionary of lists of all good and OK images for all classes
N = [1,5,10,20] # Number of top matches from database
list_of_query_images = direc_to_Oxfor_images+'query/'# './Oxford_images/query'
path_to_database = direc_to_Oxfor_images+'database/'# './Oxford_images/database'

image_norm = trsf.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
image_resize = trsf.Resize(256)
image_crop = trsf.CenterCrop((224,224))
pre_proc = trsf.Compose([
                         trsf.ToPILImage(),
                         image_resize,
                         image_crop,
                         trsf.ToTensor(),
                         image_norm
                        ])

database = FeedDB(path_to_database, transform=pre_proc)
database_loader = DataLoader(database, batch_size=1, shuffle=False)# loads './Oxford_images/database'
query_loader = DataLoader(ImageFolder(list_of_query_images,
                                      transform= trsf.Compose([
                                                 image_resize,
                                                 image_crop,
                                                 trsf.ToTensor(),
                                                 image_norm
                                                            ]))
                          , batch_size=batch_size, shuffle=False)# loads './Oxford_images/query'

# The following lines are specific to the Oxford dataset in the structure we provided.
# See README file for more details

for n in N:# Results for different top-N retrievals
    print('Case of top-'+str(n))
    with cuda.device(0):
        net.cuda()
        net.eval()
        dist_to_descrps = np.zeros((5008,5))
        with autograd.no_grad():
            precision_lists = {k: np.empty((batch_size,)) for k in query_loader.dataset.classes}
            recall_lists = {k: np.empty((batch_size,)) for k in query_loader.dataset.classes}
            for j, (im, label) in enumerate(query_loader):# extractinig the descriptor for query images
                print('Query number '+str(j)+' from class '+query_loader.dataset.classes[label[0]])
                score = np.zeros((batch_size,n))
                im = im.cuda()
                d_query = net(im)
                d_query = d_query.cpu().numpy()# VLAD descriptor of a single query
                print('search DB for matches')
                for i, X in enumerate(database_loader):  # extracting the descriptor for all DB images
                    X = X.cuda()
                    d_DB = net(X)# descriptor of a sample image from the database
                    d_DB = d_DB.cpu().numpy()
                    euclidean_dist = np.sum( (d_DB - d_query) ** 2, axis=1 )
                    dist_to_descrps[i,:] = np.reshape(euclidean_dist,(1,5))
                nearst_descrps = np.argsort(dist_to_descrps, axis=0)[:n]# Top-N retreived images
                for i in range(nearst_descrps.shape[0]):
                    retrieved = database_loader.dataset.image_list[nearst_descrps[i][0]], \
                                database_loader.dataset.image_list[nearst_descrps[i][1]], \
                                database_loader.dataset.image_list[nearst_descrps[i][2]], \
                                database_loader.dataset.image_list[nearst_descrps[i][3]], \
                                database_loader.dataset.image_list[nearst_descrps[i][4]] # name of retrieved image

                    gt_dict_key = query_loader.dataset.classes[label[0]] # Name of the class to which the query belongs
                    score [:,i] = compare_images(retrieved, gt_dict_key, list_gt) # this will have a (5,1) vector of true matches
                                                                                  # check if retrieved image is in the list
                                                                                  # of groundtruth matches of the query
                precision_per_5query = np.sum(score , axis=1)/score.shape[1]
                recall_per_5query = np.sum(score, axis=1)/len( list_gt[gt_dict_key] )
                print('Precision and recall for a query = '+\
                      str(precision_per_5query) +' and '+str(recall_per_5query))
                precision_lists[gt_dict_key] = precision_per_5query
                recall_lists[gt_dict_key] = recall_per_5query

            # Save results to file:
            file_name = 'precision_top_' + str(n) + '.npy'
            np.save(file_name, precision_lists)
            file_name = 'recall_top_' + str(n) + '.npy'
            np.save(file_name, recall_lists)

            # Computing average precision and recall:
            ave_precision = 0 # average precision over all queries
            ave_recall = 0 # average recall over all queries
            for k in precision_lists.keys():
                ave_precision += np.sum(precision_lists[k])
                ave_recall += np.sum(recall_lists[k])
            ave_precision = ave_precision / 55
            ave_recall = ave_recall / 55
            print('STOP!!')
            print("average precision is: ", ave_precision)
            print("average recall is: ", ave_recall)








