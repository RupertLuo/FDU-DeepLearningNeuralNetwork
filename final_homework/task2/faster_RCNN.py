import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(num_classes,pretrained_backbone,mask_rcnn_pretrain):
    
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,pretrained_backbone=pretrained_backbone)
    if mask_rcnn_pretrain:
        mask_rcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        model.backbone.load_state_dict(mask_rcnn.backbone.state_dict())
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model