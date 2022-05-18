import numpy as np
import cv2
import torch
import glob as glob
import os
import time
from model.faster_RCNN import create_model
import random
import numpy as np
from loguru import logger
from pathlib import Path
from configs.config_task2 import get_cfg_defaults
from data.dataset import VocCustomDataset, create_train_loader, create_valid_loader
from utils.data_utils import get_train_transform,get_valid_transform,draw_box
import torchvision
from tqdm import tqdm

def prepare_config():
    # set seeds
    cfg = get_cfg_defaults()
    cfg.merge_from_file("mid-homework/configs/experiments_task2.yaml")
    cfg.freeze()
    torch.manual_seed(cfg.RANDOM_SEED)
    random.seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    # make output dir
    logger.add(cfg.LOG_PATH,
        level='DEBUG',
        format='{time:YYYY-MM-DD HH:mm:ss} - {level} - {file} - {line} - {message}',
        rotation="10 MB")
    logger.info("Train config: %s" % str(cfg))
    model_output = Path(cfg.MODEL.saved_path)/(cfg.MODEL.name)
    model_output.mkdir(exist_ok =True)
    return cfg

def tensor2numpy(input_tensor):
	input_tensor=input_tensor.to(torch.device('cpu')).numpy()
	in_arr=np.transpose(input_tensor,(1,2,0))#将(c,w,h)转换为(w,h,c)。但此时如果是全精度模型，转化出来的dtype=float64 范围是[0,1]。后续要转换为cv2对象，需要乘以255
	return cv2.cvtColor(np.uint8(in_arr*255), cv2.COLOR_BGR2RGB)



def main(cfg):
    CLASSES = cfg.DATA.classes 
    NUM_CLASSES = cfg.DATA.num_classes
    DEVICE = cfg.TRAIN.device
    draw_proposal = True
    model_path = Path(cfg.MODEL.saved_path)/cfg.MODEL.name/'best_model.pt'
    # this will help us create a different color for each class
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # load the best model and trained weights
    model = create_model(num_classes=NUM_CLASSES)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE).eval()

    # directory where all the images are present
    test_dataset = VocCustomDataset(cfg,'test',get_valid_transform())
    
    test_loader = create_valid_loader(cfg,test_dataset)

    # define the detection threshold...
    # ... any detection having score below this will be discarded
    detection_threshold = 0.8

    # to count the total number of images iterated through
    frame_count = 0
    # to keep adding the FPS for each image
    total_fps = 0
    for i,data in enumerate(tqdm(test_loader)):
        # get the image file name for saving output later on
        img,label = data
        img = [m.to(DEVICE) for m in img]
        test_image = img[0]
        test_label = label[0]
        
        orig_image = tensor2numpy(test_image)
        orig_image = cv2.resize(orig_image, test_label['origin_shape'])
        # BGR to RGB
        
        # make the pixel range between 0 and 1
        start_time = time.time()
        with torch.no_grad():
            outputs = model(img)
        end_time = time.time()

        # get the current fps
        fps = 1 / (end_time - start_time)
        # add `fps` to `total_fps`
        total_fps += fps
        # increment frame count
        frame_count += 1
        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        # carry further only if there are detected boxes
        if not draw_proposal:
            if len(outputs[0]['boxes']) != 0:
                boxes = outputs[0]['boxes'].data.numpy()
                width,height = test_image.shape[1],test_image.shape[2]
                boxes[:,0] = (boxes[:,0]/width)*test_label['origin_shape'][0]
                boxes[:,2] = (boxes[:,2]/width)*test_label['origin_shape'][0]
                boxes[:,1] = (boxes[:,1]/height)*test_label['origin_shape'][1]
                boxes[:,3] = (boxes[:,3]/height)*test_label['origin_shape'][1]

                scores = outputs[0]['scores'].data.numpy()
                # filter out boxes according to `detection_threshold`
                boxes = boxes[scores >= detection_threshold].astype(np.int32)
                draw_boxes = boxes.copy()
                # get all the predicited class names
                pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
                
                # draw the bounding boxes and write the class name on top of it
                draw_box(orig_image,draw_boxes,pred_classes,CLASSES,COLORS,label=True)
                cv2.waitKey(1)
                cv2.imwrite(f"mid-homework/imgs/task2_output/test/{test_label['image_name']}.jpg", orig_image)
                if i >3:
                    break
        else:
            if len(outputs[0]['proposals'])!=0:
                scores = outputs[0]['proposals_score']
                sorted_score, indices = torch.sort(scores, descending=True) 
                boxes = outputs[0]['proposals'][indices][:50].data.numpy()
                
                
                width,height = test_image.shape[1],test_image.shape[2]
                boxes[:,0] = (boxes[:,0]/width)*test_label['origin_shape'][0]
                boxes[:,2] = (boxes[:,2]/width)*test_label['origin_shape'][0]
                boxes[:,1] = (boxes[:,1]/height)*test_label['origin_shape'][1]
                boxes[:,3] = (boxes[:,3]/height)*test_label['origin_shape'][1]
                draw_boxes = boxes.copy()
               
                
                # draw the bounding boxes and write the class name on top of it
                draw_box(orig_image,draw_boxes,None,CLASSES,COLORS,label=False)
                cv2.waitKey(1)
                cv2.imwrite(f"mid-homework/imgs/task2_output/proposal_box_test/{test_label['image_name']}.jpg", orig_image)
        print(f"Image {i+1} done...")
        print('-'*50)


    print('TEST PREDICTIONS COMPLETE')
    cv2.destroyAllWindows()
    # calculate and print the average FPS
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")

if __name__ == '__main__':
    cfg = prepare_config()
    main(cfg)