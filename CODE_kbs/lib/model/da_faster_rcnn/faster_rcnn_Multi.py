import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer

from model.da_faster_rcnn.DA_pooling import _ImageDA,_RoiDA
from model.da_faster_rcnn.DA_pooling import _InstanceDA
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
def BCEloss(output, label):
    loss=-(label*torch.log(output)+(1-label)*torch.log(1-output))

    return loss
def weight_attention(d):

    d = d.clamp(1e-6, 1)
    H = - (d * d.log()+ (1-d) *(1-d).log())
    return H
class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        self.RCNN_RoiDA = _RoiDA()
        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

        self.RCNN_imageDA3 = _ImageDA(256)
        self.RCNN_imageDA4 = _ImageDA(512)
        self.RCNN_imageDA = _ImageDA(self.dout_base_model)
        self.RCNN_instanceDA = _InstanceDA()
        self.consistency_loss = torch.nn.MSELoss(size_average=False)
    def forward(self, im_data, im_info, gt_boxes, num_boxes, need_backprop,
                tgt_im_data, tgt_im_info, tgt_gt_boxes, tgt_num_boxes, tgt_need_backprop):

        #assert need_backprop.detach()==1 and tgt_need_backprop.detach()==0

        batch_size = im_data.size(0)
        im_info = im_info.data     #(size1,size2, image ratio(new image / source image) )
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        need_backprop=need_backprop.data


        conv3_feat = self.conv3_s(im_data)  # ([1, 256, 150, 300])
        base_score3, base_label_cha3 = self.RCNN_imageDA3(conv3_feat)

        _,c3,w3,h3=conv3_feat.shape
        #base_score3_w=base_score3.view(1,1,w3,h3)
        w_pix3 = weight_attention(base_score3)
        w_cha3 = weight_attention(base_label_cha3)
        w_pix3_mask=5*torch.abs(w_pix3-0.6931)
        w_cha3_mask=5*torch.abs(w_cha3-0.6931)



        base_score_pix3 = Variable(torch.ones(base_score3.size(0), 1).cuda())
        DA_img_loss_cls_pix3 = (w_pix3_mask+1)*BCEloss(base_score3, base_score_pix3)
        base_score_cha3 = Variable(torch.ones(base_label_cha3.size(0), 1).cuda())
        DA_img_loss_cls_cha3 = (w_cha3_mask+1)*BCEloss(base_label_cha3, base_score_cha3)




        conv4_feat = self.conv34_s(conv3_feat)  # ([1, 512, 75, 150])
        base_score4, base_label_cha4 = self.RCNN_imageDA4(conv4_feat)
        base_score_pix4 = Variable(torch.ones(base_score4.size(0), 1).cuda())
        DA_img_loss_cls_pix4 = BCEloss(base_score4, base_score_pix4)
        base_score_cha4 = Variable(torch.ones(base_label_cha4.size(0), 1).cuda())
        DA_img_loss_cls_cha4 = BCEloss(base_label_cha4, base_score_cha4)


        base_feat = self.conv45_s(conv4_feat)  # ([1, 512, 37, 75])
        base_score5, base_label_cha5 = self.RCNN_imageDA(base_feat)
        base_score_pix5 = Variable(torch.ones(base_score5.size(0), 1).cuda())
        DA_img_loss_cls_pix5 = BCEloss(base_score5, base_score_pix5)
        base_score_cha5 = Variable(torch.ones(base_label_cha5.size(0), 1).cuda())
        DA_img_loss_cls_cha5 = BCEloss(base_label_cha5, base_score_cha5)


        DA_img_loss_cls_pix = DA_img_loss_cls_pix3.mean() + DA_img_loss_cls_pix4.mean() + DA_img_loss_cls_pix5.mean()
        DA_img_loss_cls_cha = DA_img_loss_cls_cha3.mean() + DA_img_loss_cls_cha4.mean() + DA_img_loss_cls_cha5.mean()



        # feed base feature map tp RPN to obtain rois
        self.RCNN_rpn.train()
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))


        pooled_feat = self._head_to_tail(pooled_feat)
        source_ture=pooled_feat
        source_virtual=pooled_feat
        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob2 = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob2.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        """ =================== for target =========================="""

        tgt_batch_size = tgt_im_data.size(0)
        tgt_im_info = tgt_im_info.data  # (size1,size2, image ratio(new image / source image) )
        tgt_gt_boxes = tgt_gt_boxes.data
        tgt_num_boxes = tgt_num_boxes.data
        #tgt_need_backprop = tgt_need_backprop.data


        tgt_conv3_feat = self.conv3_s(tgt_im_data)
        tgt_base_score3, tgt_base_label_cha3 = self.RCNN_imageDA3(tgt_conv3_feat)

        _, t_c3, t_w3, t_h3 = tgt_conv3_feat.shape
        tgt_w_pix3 = weight_attention(tgt_base_score3)
        tgt_w_cha3 = weight_attention(tgt_base_label_cha3)
        tgt_w_pix3_mask = 5 * torch.abs(tgt_w_pix3- 0.6931)
        tgt_w_cha3_mask = 5 * torch.abs(tgt_w_cha3 - 0.6931)

        tgt_base_score_pix3 = Variable(torch.zeros(tgt_base_score3.size(0), 1).cuda())
        tgt_DA_img_loss_cls_pix3 = (1+tgt_w_pix3_mask)*BCEloss(tgt_base_score3, tgt_base_score_pix3)
        tgt_base_score_cha3 = Variable(torch.zeros(tgt_base_label_cha3.size(0), 1).cuda())
        tgt_DA_img_loss_cls_cha3 = (1+tgt_w_cha3_mask)*BCEloss(tgt_base_label_cha3, tgt_base_score_cha3)






        tgt_conv4_feat = self.conv34_s(tgt_conv3_feat)
        tgt_base_score4, tgt_base_label_cha4 = self.RCNN_imageDA4(tgt_conv4_feat)
        tgt_base_score_pix4 = Variable(torch.zeros(tgt_base_score4.size(0), 1).cuda())
        tgt_DA_img_loss_cls_pix4 = BCEloss(tgt_base_score4, tgt_base_score_pix4)
        tgt_base_score_cha4 = Variable(torch.zeros(tgt_base_label_cha4.size(0), 1).cuda())
        tgt_DA_img_loss_cls_cha4 = BCEloss(tgt_base_label_cha4, tgt_base_score_cha4)


        tgt_base_feat = self.conv45_s(tgt_conv4_feat)
        tgt_base_score5, tgt_base_label_cha5 = self.RCNN_imageDA(tgt_base_feat)
        tgt_base_score_pix5 = Variable(torch.zeros(tgt_base_score5.size(0), 1).cuda())
        tgt_DA_img_loss_cls_pix5 =BCEloss(tgt_base_score5, tgt_base_score_pix5)
        tgt_base_score_cha5 = Variable(torch.zeros(tgt_base_label_cha5.size(0), 1).cuda())
        tgt_DA_img_loss_cls_cha5 =BCEloss(tgt_base_label_cha5, tgt_base_score_cha5)


        tgt_DA_img_loss_cls_pix = tgt_DA_img_loss_cls_pix3.mean() + tgt_DA_img_loss_cls_pix4.mean() + tgt_DA_img_loss_cls_pix5.mean()
        tgt_DA_img_loss_cls_cha = tgt_DA_img_loss_cls_cha3.mean() + tgt_DA_img_loss_cls_cha4.mean() + tgt_DA_img_loss_cls_cha5.mean()





        # feed base feature map tp RPN to obtain rois
        self.RCNN_rpn.eval()
        tgt_rois, tgt_rpn_loss_cls, tgt_rpn_loss_bbox = \
            self.RCNN_rpn(tgt_base_feat, tgt_im_info, tgt_gt_boxes, tgt_num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining

        tgt_rois_label = None
        tgt_rois_target = None
        tgt_rois_inside_ws = None
        tgt_rois_outside_ws = None
        tgt_rpn_loss_cls = 0
        tgt_rpn_loss_bbox = 0

        tgt_rois = Variable(tgt_rois)
        # do roi pooling based on predicted rois

        if  cfg.POOLING_MODE == 'align':
            tgt_pooled_feat = self.RCNN_roi_align(tgt_base_feat, tgt_rois.view(-1, 5))
        tgt_pooled_feat = self._head_to_tail(tgt_pooled_feat)
        target_ture=tgt_pooled_feat
        target_virtual=tgt_pooled_feat



        source_ins_ture,source_ins_virtual = self.RCNN_instanceDA(source_ture,source_virtual)
        source_ture_DA_ins=nn.BCELoss()(source_ins_ture,torch.ones_like(source_ins_ture))
        source_virtual_DA_ins = nn.BCELoss()(source_ins_virtual, torch.zeros_like(source_ins_virtual))

        target_ins_ture, target_ins_virtual = self.RCNN_instanceDA(target_ture, target_virtual)
        target_ture_DA_ins = nn.BCELoss()(target_ins_ture,torch.zeros_like(target_ins_ture))
        target_virtual_DA_ins = nn.BCELoss()(target_ins_virtual,torch.ones_like(target_ins_virtual))

        ture_loss=target_ture_DA_ins.mean()+source_ture_DA_ins.mean()
        virtual_loss=target_virtual_DA_ins.mean()+source_virtual_DA_ins.mean()



        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, \
               DA_img_loss_cls_pix,DA_img_loss_cls_cha,tgt_DA_img_loss_cls_pix,tgt_DA_img_loss_cls_cha,ture_loss,virtual_loss


    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
