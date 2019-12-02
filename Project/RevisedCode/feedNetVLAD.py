import os
import torch
import numpy as np
import random
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class DataFeedTriple(Dataset):
    '''
    Generating triplet data-samples to feed to NetVLAD during training. A sample is
    composed of three images, a query, a match, and a non-match. It assumes all images
    are divided into three sub-directories named: query, positives, and negatives.
    '''
    def __init__(self,root_dir, dir1='query',dir2='positive',dir3='negative',
                 transform=None):
        self.root_dir = root_dir
        self.query = dir1
        self.positive = dir2
        self.negative = dir3
        self.transform = transform
        self.query_list = sorted(os.listdir(self.root_dir + '/' + self.query))
        self.positive_list = sorted(os.listdir(self.root_dir + '/' + self.positive))
        self.negative_list = sorted(os.listdir(self.root_dir + '/' + self.negative))
        # Randomly shuffle the triplets before reading
        zipped_lists = list( zip(self.query_list, self.positive_list,
                                 self.negative_list) )
        random.shuffle(zipped_lists)
        self.query_list, self.positive_list, self.negative_list = zip(*zipped_lists)


    def __len__(self):
        return len( self.query_list )

    def __getitem__(self, idx):

        query = io.imread(self.root_dir+'/'+self.query+'/'+self.query_list[idx])# Read one image indexed by idx
        positive = io.imread(self.root_dir+'/'+self.positive+'/'+self.positive_list[idx])
        negative = io.imread(self.root_dir+'/'+self.negative+'/'+self.negative_list[idx])
        if self.transform:
            query = self.transform(query)
            positive = self.transform(positive)
            negative = self.transform(negative)
        return (query, positive, negative)

class FeedDB(Dataset):
    '''
    Prepares database images to be fed to NetVLAD during the validation phase
    '''
    def __init__(self,dir_to_DB, transform=None):
        self.dir_DB = dir_to_DB
        self.transform = transform
        self.image_list = sorted(os.listdir(self.dir_DB))# List of image names

    def __len__(self):
        return len( self.image_list )

    def __getitem__(self, idx):
        image = io.imread(self.dir_DB+self.image_list[idx])# Read one image indexed by idx
        if self.transform:# Preprocess image
            image = self.transform(image)
        return image


