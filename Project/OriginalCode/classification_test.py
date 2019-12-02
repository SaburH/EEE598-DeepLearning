'''
The main script for building a pretrained resnet-50 model and test it on the
ILSVRC2012 validation set.
The reported accuracy is 76.15%.
'''

import torch
import torch.cuda as gpu
import torch.autograd as autgrad
import torch.utils.model_zoo
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from buildNetwork import resnet50

downloaded_data ="./ILSVRC2012_img_val/" # please set this variable to where you downloaded the dataset

###############################  Helper functions  ###################################
# dataLoader prepares and loads ILSVRC2012 validation data. See the README file for
# more details
def dataLoader(imagMean, imagStd, scale=256, cropSize=224, batchSize=50):
    scaleImag = transforms.Resize(scale)
    cropImag = transforms.CenterCrop((cropSize, cropSize))
    normImag = transforms.Normalize(mean=imagMean, std=imagStd)

    T = transforms.Compose([scaleImag,
                            cropImag,
                            transforms.ToTensor(),
                            normImag])

    dataLoader = DataLoader(datasets.ImageFolder(downloaded_data, T), batch_size=batchSize)
    return dataLoader
#####################################################################################


# Build model:
# ------------
net = resnet50(pretrained=True)
net.cuda()
net.eval()

# Get dataset:
# ------------
loader = dataLoader([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])


# Test model:
# -----------

with autgrad.no_grad():
    accAccuracy = []
    for image, labels in loader:
        inputs = image.cuda()
        pred = net(inputs)
        _, predLabels = torch.max(pred ,1)
        predLabels = predLabels.cpu().numpy()
        l = labels.cpu().numpy()
        accAccuracy.append( (l == predLabels) )
        accuracyPerBatch = np.sum( (l == predLabels) )/50
        print("Per Batch Accuracy= ",accuracyPerBatch)
    aveAccuracy = np.sum( accAccuracy )/50000
    print('Pre-trained model accuracy = '+ str(aveAccuracy))
