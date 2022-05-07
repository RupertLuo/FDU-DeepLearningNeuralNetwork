import numpy as np
import cv2
import torch
import glob as glob
import os
import time
from model.YOLO import Yolov3
from utils.Decode_Box import DecodeBox
import random
import numpy as np
from PIL import ImageDraw, ImageFont
from loguru import logger
from pathlib import Path
from configs.config_task2 import get_cfg_defaults
from data.dataset import VocCustomDataset, create_train_loader, create_valid_loader
from utils.data_utils import get_train_transform,get_valid_transform,draw_box
import torchvision
import colorsys
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
    model = Yolov3(cfg)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE).eval()

    # directory where all the images are present
    test_dataset = VocCustomDataset(cfg,'test',get_valid_transform())
    
    test_loader = create_valid_loader(cfg,test_dataset)
    bbox_util = DecodeBox(cfg.MODEL["ANCHORS"], num_classes = len(cfg.DATA["classes"]), input_shape = (cfg.DATA["width"], cfg.DATA["height"]), anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    # define the detection threshold...
    # ... any detection having score below this will be discarded
    detection_threshold = 0.8

    # to count the total number of images iterated through
    frame_count = 0
    # to keep adding the FPS for each image
    total_fps = 0
    
    VOC = False
    if VOC:
        for i,data in enumerate(tqdm(test_loader)):
            # get the image file name for saving output later on
            img,label = data
            img = [m.to(DEVICE) for m in img]
            test_image = img[0]
            test_label = label[0]
            
            orig_image = tensor2numpy(test_image)
            orig_image = cv2.resize(orig_image, test_label['origin_shape'])
            # BGR to RGB
            images_list = []
            for im in img:
                images_list.append(im.unsqueeze(0))
            images_list = torch.cat(images_list)
            
            # make the pixel range between 0 and 1
            start_time = time.time()
            image_shape = np.array(np.shape(img[0])[1:])
            with torch.no_grad():
                results = model(images_list)
                results = bbox_util.decode_box(results)
                outputs = bbox_util.non_max_suppression(torch.cat(results, 1), num_classes = len(cfg.DATA["classes"]), input_shape = (cfg.DATA["width"], cfg.DATA["height"]), 
                            image_shape = image_shape, letterbox_image = False, conf_thres = detection_threshold, nms_thres = 0.3)
            
            end_time = time.time()
            
            boxes =  []
            pred_classes = []
            scores = []
            try:
            # top, left, bottom, right
                for j in range(outputs[0].shape[1]):
                    output = outputs[0]
                    width,height = test_image.shape[1],test_image.shape[2]
                    output[j,0] = max((output[j,0]/width)*test_label['origin_shape'][1], 0)
                    output[j,1] = max((output[j,1]/width)*test_label['origin_shape'][0], 0)
                    output[j,2] = min((output[j,2]/height)*test_label['origin_shape'][1], test_label['origin_shape'][1])
                    output[j,3] = min((output[j,3]/height)*test_label['origin_shape'][0], test_label['origin_shape'][0])
                    boxes.append([output[j,1], output[j,0], output[j,3], output[j,2]])
                    scores.append(output[j,5])
                    pred_classes.append(CLASSES[int(output[j,6])])
                    if j > 3:
                        break
                draw_box(orig_image,boxes,pred_classes,scores,CLASSES,COLORS,label=True)
                cv2.waitKey(1)
                cv2.imwrite(f"mid-homework/imgs/task2_output/yolo/{test_label['image_name']}.jpg", orig_image)
            except:
                pass
            # get the current fps
            fps = 1 / (end_time - start_time)
            # add `fps` to `total_fps`
            total_fps += fps
            # increment frame count
            frame_count += 1

        print('TEST PREDICTIONS COMPLETE')
        cv2.destroyAllWindows()
        # calculate and print the average FPS
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")

    else:
        detection_threshold = 0.7
        image_dir = [p for p in Path('mid-homework/imgs/task3/input').rglob("*.jpg")]
        for img_path in image_dir:
            image = cv2.imread(str(img_path))
            orig_image = image.copy()
            # BGR to RGB
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)

            # make the pixel range between 0 and 1
            image /= 255.0
            image = cv2.resize(image, (416, 416))
            # bring color channels to front
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            # convert to tensor
            image = torch.tensor(image, dtype=torch.float).to(DEVICE)
            # add batch dimension
            
            image = torch.unsqueeze(image, 0)
            image_shape = np.array(np.shape(image[0])[1:])
            with torch.no_grad():
                results = model(image.to(DEVICE))
                results = bbox_util.decode_box(results)
                outputs = bbox_util.non_max_suppression(torch.cat(results, 1), num_classes = len(cfg.DATA["classes"]), input_shape = (cfg.DATA["width"], cfg.DATA["height"]), 
                            image_shape = image_shape, letterbox_image = False, conf_thres = detection_threshold, nms_thres = 0.4)
            
            end_time = time.time()
            
            boxes =  []
            pred_classes = []
            scores = []
            
            # top, left, bottom, right
            try:
                for j in range(outputs[0].shape[0]):
                    output = outputs[0]
                    width,height = image_shape
                    output[j,0] = max((output[j,0]/width)*orig_image.shape[0], 0)
                    output[j,1] = max((output[j,1]/width)*orig_image.shape[1], 0)
                    output[j,2] = min((output[j,2]/height)*orig_image.shape[0], orig_image.shape[0])
                    output[j,3] = min((output[j,3]/height)*orig_image.shape[1], orig_image.shape[1])
                    boxes.append([output[j,1], output[j,0], output[j,3], output[j,2]])
                    scores.append(output[j,5] * output[j,4])
                    pred_classes.append(CLASSES[int(output[j,6])])
                draw_box(orig_image,boxes,pred_classes,scores,CLASSES,COLORS,label=True)
                cv2.waitKey(1)
                cv2.imwrite(f"mid-homework/imgs/task3/output/Yolo/{img_path.name}", orig_image)
            except:
                pass
if __name__ == '__main__':
    cfg = prepare_config()
    main(cfg)