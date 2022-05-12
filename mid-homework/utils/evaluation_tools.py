import torch
from podm.metrics import get_pascal_voc_metrics, MetricPerClass, get_bounding_boxes,BoundingBox
from podm.metrics import MetricPerClass
from podm.box import Box, intersection_over_union
import numpy as np
def calculate_iou(box_a, box_b):
    # box_a: (num, 4) ----- number of boxes, (x1y1x2y2)
    '''
    #-----------------------------------------------------------#
    #   计算真实框的左上角和右下角
    #-----------------------------------------------------------#
    b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
    b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
    #-----------------------------------------------------------#
    #   计算先验框获得的预测框的左上角和右下角
    #-----------------------------------------------------------#
    b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
    b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2
    
    #-----------------------------------------------------------#
    #   将真实框和预测框都转化成左上角右下角的形式
    #   box_a: (num, 4) ----- number of boxes, (x1y1x2y2)
    #-----------------------------------------------------------#
    box_a = torch.zeros_like(_box_a)
    box_b = torch.zeros_like(_box_b)
    box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
    box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2
    '''
    #-----------------------------------------------------------#
    #   A为真实框的数量，B为先验框的数量
    #-----------------------------------------------------------#
    A = box_a.size(0)
    B = box_b.size(0)

    #-----------------------------------------------------------#
    #   计算交的面积
    #-----------------------------------------------------------#
    max_xy  = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy  = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter   = torch.clamp((max_xy - min_xy), min=0)
    inter   = inter[:, :, 0] * inter[:, :, 1]
    #-----------------------------------------------------------#
    #   计算预测框和真实框各自的面积
    #-----------------------------------------------------------#
    area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    #-----------------------------------------------------------#
    #   求IOU
    #-----------------------------------------------------------#
    union = area_a + area_b - inter
    iou = torch.diag(inter / union)
    #----------------------------------------------------#
    #   找到包裹两个框的最小框的左上角和右下角
    #----------------------------------------------------#
    enclose_mins    = torch.min(box_a[:, :2], box_b[:, :2])
    enclose_maxes   = torch.max(box_a[:, 2:], box_b[:, 2:])
    enclose_wh      = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(enclose_maxes))
    #----------------------------------------------------#
    #   计算对角线距离
    #----------------------------------------------------#
    enclose_area    = enclose_wh[..., 0] * enclose_wh[..., 1]
    giou            = iou - (enclose_area - torch.diag(union)) / enclose_area
    
    return iou, giou  # [A,B]
def get_mIOU(pre_boxes,target_boxes,iou_threshold):
    mIOU_list_each = []
    matched_target = []
    for k in range(len(pre_boxes)):
        for j in range(len(target_boxes)):
            box1 = Box.of_box(pre_boxes[k][0], pre_boxes[k][1], pre_boxes[k][2], pre_boxes[k][3])
            box2 = Box.of_box(target_boxes[j][0], target_boxes[j][1], target_boxes[j][2], target_boxes[j][3])
            IOU = intersection_over_union(box1, box2)
            if IOU >= iou_threshold and j not in matched_target:
                matched_target.append(j)
                mIOU_list_each.append(IOU)
    mIOU = sum(mIOU_list_each) / (len(mIOU_list_each)+1e-5)
    return mIOU
def get_mAP(pre_class,pre_boxes,pre_scores,target_class,target_boxes,iou_threshold):
    '''
    pre_class = [1,4,3]
    pre_boxes = [[10, 20, 100, 110], [5, 24, 200,250], [125, 164, 205, 175]]
    pre_scores = [0.4,0.5,0.9]
    taget_class = [1,3,4]
    target_boxes = [[20, 20, 110, 110], [25, 5, 250,200], [145, 162, 199, 205]]
    '''
    pre_box = []
    for i in range(len(pre_class)):
        bb = BoundingBox.of_bbox(None,category = pre_class[i],xtl = pre_boxes[i][0],ytl = pre_boxes[i][1],xbr = pre_boxes[i][2], ybr = pre_boxes[i][3],score = pre_scores[i])
        pre_box.append(bb)
    
    target_box = []
    for i in range(len(target_class)):
        bb = BoundingBox.of_bbox(None,category = target_class[i],xtl = target_boxes[i][0],ytl = target_boxes[i][1],xbr = target_boxes[i][2], ybr = target_boxes[i][3])
        target_box.append(bb)
    #----------------------------------------------------#
    #   计算mAP
    #----------------------------------------------------#
    results = get_pascal_voc_metrics(target_box, pre_box, iou_threshold)
    tps_list = []
    for cls, metric in results.items():
        tps_list.append(metric.tp)
    tp = sum(tps_list)/len(tps_list)
    acc = tp / (len(pre_class)+1e-5)
    mAP = MetricPerClass.mAP(results)
    return mAP,acc






    
if __name__ == "__main__":
    pre = [[10, 20, 100, 110], [5, 24, 200,250], [125, 164, 205, 175]]
    target = [[20, 20, 110, 110], [25, 5, 250,200], [145, 162, 199, 205]]
    taget_class = [1,3,4]
    pre_class = [1,4,3]
    mAP = get_mAP(pre_class,pre,taget_class,target)
    print(mAP)
