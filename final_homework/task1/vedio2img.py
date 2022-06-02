import cv2
import os
import time
import datetime
import random

"""
 Read video and convert it into pictures
 读取视频并将视频转换为图片
 注意请勿在视频文件夹中放其他文件
"""

def save_img():
    # video_path = r'E:/VideoDir/'
    # videos = os.listdir(video_path)
    videos = load_video(r'final_homework/task1/data/vedio/if_photos_gif.mp4')
    for video_name in videos:
        vc = cv2.VideoCapture(video_name)  # 读入视频文件
        c = 1
        if vc.isOpened():  # 判断是否正常打开
            rval, frame = vc.read()
        else:
            rval = False
        timeF = 1  # 视频帧计数间隔频率
        i = 0
        while rval:  # 循环读取视频帧
            rval, frame = vc.read()
            if (c % timeF == 0):  # 每隔timeF帧进行存储操作
                if frame is not None:
                    cv2.imwrite("final_homework/task1/data/images/" + str(c) + '_' + str(random.randint(1, 100000)) +
                            '.jpg', frame)  
            c = c + 1
            cv2.waitKey(1)
        vc.release()

def load_video(video_path):
    input_path_extension = video_path.split('.')[-1]
    if input_path_extension in ['mp4']:
        return [video_path]
    elif input_path_extension == "txt":
        with open(video_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(os.path.join(video_path, "*.mp4"))

if __name__ == '__main__':
    save_img()