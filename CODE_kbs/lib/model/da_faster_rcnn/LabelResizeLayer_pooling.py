
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from lib.model.utils.config import cfg
import torch.nn as nn
from torch.autograd import Function
import cv2

def flat(nums):  # 将嵌套列表变为一层list
    res = []
    for i in nums:
        if isinstance(i, list):
            res.extend(flat(i))
        else:
            res.append(i)
    return res



class ImageLabelResizeLayer(nn.Module):
    """
    Resize label to be the same size with the samples
    """
    def __init__(self):
        super(ImageLabelResizeLayer, self).__init__()


    def forward(self,x_pix,need_backprop):
        feats_pix = x_pix.detach().cpu().numpy()

        lbs = need_backprop.detach().cpu().numpy()
        gt_blob_pix = np.zeros((lbs.shape[0], feats_pix.shape[2], feats_pix.shape[3], 1), dtype=np.float32)

        for i in range(lbs.shape[0]):
            lb = np.array([lbs[i]])
            lbs_resize_pix = cv2.resize(lb, (feats_pix.shape[3], feats_pix.shape[2]), interpolation=cv2.INTER_NEAREST)

            gt_blob_pix[i, 0:lbs_resize_pix.shape[0], 0:lbs_resize_pix.shape[1], 0] = lbs_resize_pix


        channel_swap_pix = (0, 3, 1, 2)

        gt_blob_pix = gt_blob_pix.transpose(channel_swap_pix)

        y_pix = Variable(torch.from_numpy(gt_blob_pix)).cuda()
        y_pix = y_pix.squeeze(1).long()



        return y_pix


class InstanceLabelResizeLayer(nn.Module):


    def __init__(self):
        super(InstanceLabelResizeLayer, self).__init__()
        self.minibatch=256

    def forward(self, x,need_backprop):
        feats = x.data.cpu().numpy()
        lbs = need_backprop.data.cpu().numpy()
        resized_lbs = np.ones((feats.shape[0], 1), dtype=np.float32)
        for i in range(lbs.shape[0]):
            resized_lbs[i*self.minibatch:(i+1)*self.minibatch] = lbs[i]

        y=torch.from_numpy(resized_lbs).cuda()

        return y
