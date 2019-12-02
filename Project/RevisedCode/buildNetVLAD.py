'''
NetVLAD implementation.
NetVLAD is a network trained to extract a Global Descriptor (GD) for any
input image. Details of the network could be found in the following paper:
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7937898

This implementation script is inspired by the one found on:
https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from buildNetwork import resnet50


class NetVLAD(nn.Module):
    '''
    This class builds the VLAD layer and connects it to the feature extractor
    '''

    def __init__(self, dim, base_model = 'resnet50',num_clusters=64, alpha=1.0,
                 normalize_input=True, endToEnd = False):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.normalize_input = normalize_input
        self.alpha = alpha
        c = torch.empty(num_clusters, dim)
        self.centroids = nn.Parameter( nn.init.normal_(c, mean=0, std= 0.01) )
        self.conv = nn.Conv2d(self.dim, self.num_clusters, kernel_size=(1, 1), bias=True)
        self.link_params()
        self.base_model = base_model
        if self.base_model is 'resnet50': # Other nets could be implemented
            init_baseNET = resnet50(pretrained=True)
            layers = init_baseNET.children()
            listLayers = list(layers)
            self.baseNet = nn.Sequential(*listLayers[:-2])# Discard average pooling and FC layers

        if not endToEnd:# Option for training end-to-end
            for param in self.baseNet.parameters():
                param.requires_grad = False

    def link_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )# Relating the convolutional
         # layer weights to the centroids
         # of the dictionary
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):

        x = self.baseNet(x)# output size: B x C x H x W
        B, C = x.shape[:2]# B is batchSize, C is number of input channels

        '''''''''''''''''''''''''''''''''''
          VLAD module starts from here
          
        '''''''''''''''''''''''''''''''''''
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)# across descriptor dimension

        # Soft Cluster Assignment
        projDesc = self.conv(x)# B x K x H x W --> K = numClusters
        projDesc = projDesc.view(B,self.num_clusters, -1)# flatten the spatial dimensions: B x K x N --> N = H x W
        soft_assign = F.softmax(projDesc, dim=1)# produces a prob. on descriptors assignment to centroids
                                                # B x K x N

        xFlatten = x.view(B, C, -1)# Flatten spatial dimensions: B x C x N

        # Calculate Residuals to Each Cluster
        xRep = xFlatten.expand(self.num_clusters, -1, -1, -1)# K copies of xFlatten with size: B x C x N --> N = H x W
        xRep = xRep.permute(1, 0, 2, 3)# output: B x K x C x N--> prepare to subtract from Centroid matrix
        centroidRep = self.centroids.expand(xFlatten.size(-1), -1, -1)# N copies of centroids with size: K x C
        centroidRep = centroidRep.permute(1, 2, 0).unsqueeze(0)# 1 x K x C x N
        residual = xRep - centroidRep# Subtract centroidRep from all descriptors in one batch
                                     # B x K x C x N
        residual *= soft_assign.unsqueeze(2)# multiplying soft-assign weights with the residuals
                                            # soft_assign has size: B x K x 1 x N
        vlad = residual.sum(dim=-1)# VLAD matrix: B x K x C

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize. Final GD of size: B x (KC)

        return vlad

