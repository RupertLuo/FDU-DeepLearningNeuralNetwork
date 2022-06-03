from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image

from colorize import Colorize
from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path
import torchvision
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
vedio_img_dir = 'final_homework/task1/data/images'
vedio_img_list = sorted(list(Path(vedio_img_dir).glob('*.jpg')))
colorize = Colorize(19)
for i,img_path in enumerate(tqdm(vedio_img_list)):
    image = Image.open(img_path).convert('RGB')
    oh,ow = image.size
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits[0] 
    mask = F.softmax(logits, dim=0).max(0).indices
    color_pic = colorize(mask.cpu())
    pic= torchvision.transforms.functional.to_pil_image(color_pic)
    pic = pic.resize((int(oh),int(ow)))
    pic.save('final_homework/task1/data/image_seg/'+img_path.name)

 # shape (batch_size, num_labels, height/4, width/4)

