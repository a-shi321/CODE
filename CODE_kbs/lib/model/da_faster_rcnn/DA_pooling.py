from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
from lib.model.da_faster_rcnn.LabelResizeLayer_pooling import ImageLabelResizeLayer
from lib.model.da_faster_rcnn.LabelResizeLayer_pooling import InstanceLabelResizeLayer



class GRLayer(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.alpha=0.1

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output=grad_outputs.neg() * ctx.alpha
        return output

def grad_reverse(x):
    return GRLayer.apply(x)

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)


class _ImageDA(nn.Module):
    def __init__(self,dim):
        super(_ImageDA,self).__init__()
        self.dim=dim  # feat layer          256*H*W for vgg16
        # 像素考虑
        self.Conv1_pix = nn.Conv2d(self.dim, 512, kernel_size=1, stride=1, bias=False)
        self.bn_pix1 = nn.BatchNorm2d(512, affine=True)
        self.Conv2_pix = nn.Conv2d(512, 1, kernel_size=1, stride=1, bias=False)
        self.reLu_pix = nn.ReLU(inplace=False)
        self.sigmoid_pix = nn.Sigmoid()




        # 通道考虑
        self.Conv1_cha = nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv2_cha = nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_cha = nn.BatchNorm2d(self.dim, affine=True)
        self.reLu_cha = nn.ReLU(inplace=False)
        self.avg_pool_cha = nn.AdaptiveAvgPool2d(1)
        self.sigmoid_cha = nn.Sigmoid()



    def forward(self,x):
        x = grad_reverse(x)
        #像素对齐
        x_pix = self.reLu_pix(self.bn_pix1(self.Conv1_pix(x)))
        x_pix = self.Conv2_pix(x_pix)
        x_pix = self.sigmoid_pix(x_pix)
        x_pix = x_pix.view(-1, 1)

        #print("x_pix", x_pix)

        #通道对齐
        x_cha = self.reLu_cha(self.bn_cha(self.Conv1_cha(x)))
        x_cha = self.Conv2_cha(x_cha) # (1,256,37,75)
        x_cha = self.avg_pool_cha(x_cha)  # 这里变为（1,512,1,1）
        x_cha = self.sigmoid_cha(x_cha)
        x_cha = x_cha.view(-1, 1)
        # print("x_cha", x_cha.shape)
        #print("x_cha", x_cha)
        return x_pix,x_cha

class _RoiDA(nn.Module):
    def __init__(self):
        super(_RoiDA,self).__init__()

        self.Conv_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn= nn.BatchNorm2d(512, affine=True)
        self.reLu = nn.ReLU(inplace=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.dc_ip1 = nn.Linear(512, 128)
        self.dc_relu1 = nn.ReLU()

        self.dc_ip2 = nn.Linear(128, 128)
        self.dc_relu2 = nn.ReLU()
        self.clssifer = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_cls,x_reg):

        x_cls_grl=grad_reverse(x_cls)
        x_cls_grl=F.relu(self.bn(self.Conv_1(x_cls_grl)))
        x_cls_grl=F.relu(self.bn(self.Conv_2(x_cls_grl)))
        x_cls_grl=self.avg_pool(x_cls_grl)

        x_cls_grl=x_cls_grl.view(-1,512)
        x_cls_grl = self.dc_relu1(self.dc_ip1(x_cls_grl))
        x_cls_grl = self.dc_relu2(self.dc_ip2(x_cls_grl))
        x_cls_grl = self.clssifer(x_cls_grl)
        x_cls_grl = self.sigmoid(x_cls_grl)

        x_reg_grl = grad_reverse(x_reg)
        x_reg_grl = F.relu(self.bn(self.Conv_1(x_reg_grl)))
        x_reg_grl = F.relu(self.bn(self.Conv_2(x_reg_grl)))
        x_reg_grl = self.avg_pool(x_reg_grl)

        x_reg_grl = x_reg_grl.view(-1, 512)
        x_reg_grl = self.dc_relu1(self.dc_ip1(x_reg_grl))
        x_reg_grl = self.dc_relu2(self.dc_ip2(x_reg_grl))
        x_reg_grl = self.clssifer(x_reg_grl)
        x_reg_grl = self.sigmoid(x_reg_grl)

        return x_cls_grl,x_reg_grl



class _InstanceDA(nn.Module):
    def __init__(self):
        super(_InstanceDA,self).__init__()
        self.dc_ip1 = nn.Linear(4096, 1024)
        self.dc_relu1 = nn.ReLU()
        self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(1024, 1024)
        self.dc_relu2 = nn.ReLU()
        self.dc_drop2 = nn.Dropout(p=0.5)

        self.clssifer=nn.Linear(1024,1)
        self.LabelResizeLayer=InstanceLabelResizeLayer()

    def forward(self,x_ture,x_virtual):
        x_ture=grad_reverse(x_ture)
        x_ture=self.dc_drop1(self.dc_relu1(self.dc_ip1(x_ture)))
        x_ture=self.dc_drop2(self.dc_relu2(self.dc_ip2(x_ture)))
        x_ture=F.sigmoid(self.clssifer(x_ture))

        x_virtual = grad_reverse(x_virtual)
        x_virtual = self.dc_drop1(self.dc_relu1(self.dc_ip1(x_virtual)))
        x_virtual = self.dc_drop2(self.dc_relu2(self.dc_ip2(x_virtual)))
        x_virtual = F.sigmoid(self.clssifer(x_virtual))

        return x_ture,x_virtual


