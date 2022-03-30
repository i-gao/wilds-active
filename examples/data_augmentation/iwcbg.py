import pandas as pd
import torch
import json
from PIL import Image, ImageDraw, ImageFilter
import types
import os

def xywh_to_xyxy(x, y, w, h, xy_center=False):
    """convert [x,y,w,h] -> [x_1, y_1, x_2, y_2]"""
    if xy_center: 
        # (x,y) gives the center of the box
        x1, y1 = x-w/2, y-h/2
        x2, y2 = x+w/2, y+h/2
    else: 
        # (x,y) gives the upper left corner of the box
        x1, y1 = x, y
        x2, y2 = x+w, y+h
    return int(x1), int(y1), int(x2), int(y2)

def xyxy_to_xywh(x1, y1, x2, y2, xy_center=False):
    """convert [x_1, y_1, x_2, y_2] -> [x,y,w,h]"""
    if xy_center: 
        # (x, y) gives the center of the box
        x, y = (x1+x2)/2, (y1+y2)/2
    else: 
        # (x,y) gives the upper left corner of the box
        x, y = min(x1, x2), min(y1, y2)        
    w = abs(x2-x1)
    h = abs(y2-y1)
    return int(x), int(y), int(w), int(h)

class iWildCamBackgroundAugment:
    def __init__(self, pad=50):
        self.pad = pad
    
    def __call__(self, img, bboxes, conf, empty):
        """paste the animal in the bbox of full_ds[ix] onto an empty image"""
        # 1. get animal
        if len(bboxes) == 0 or conf < 0.5: return img
        img_w, img_h = img.width, img.height
        
        # 2. get background
        bg_img = empty.copy()
        bg_img = bg_img.resize(img.size)
        
        # 3. copy paste animals onto new bg   
        for bbox in bboxes:
            x,y,w,h = bbox
            mask_im = Image.new("L", img.size, 0) # make a mask & fill with black
            draw = ImageDraw.Draw(mask_im) # make canvas
            xyxy = xywh_to_xyxy(
                x * img_w - self.pad/2,  # fraction -> pixels
                y * img_h - self.pad/2, 
                w * img_w + self.pad, 
                h * img_h + self.pad,
            )
            draw.rectangle(xyxy, fill=255) # draw a rectangle of white where bbox is
            mask_im = mask_im.filter(ImageFilter.GaussianBlur(5))
            bg_img.paste(
                img,
                box=(0,0), # paste animal in same place as in og img
                mask=mask_im
            ) 
        return bg_img
       
def modify_getitem_fn_for_iwctransform(subset):
    subset.aug = iWildCamBackgroundAugment()
    subset.img_ids = pd.read_csv(subset._data_dir / 'metadata.csv')['image_id']
    subset.bbox_df = json.load(open(f"{os.path.dirname(os.path.realpath(__file__))}/megadetector_results.json", 'r'))['images'] # json stored in examples/data_augmentation
    subset.bbox_df = pd.DataFrame(subset.bbox_df).set_index('id')

    def new_fn(self, idx):
        x, y, metadata = self.dataset[self.indices[idx]]
        if y != 0:
            empty = self.dataset[torch.random.choice(self.dataset.indices[self.dataset.y_array == 0])]
            # get bboxes, conf
            image_id = self.img_ids.loc[idx]
            bboxes, conf = self.bbox_df.loc[image_id]['detections'], self.bbox_df.loc[image_id]['max_detection_conf']
            bboxes = [b['bbox'] for b in bboxes]
            x = self.aug(x, bboxes, conf, empty)
        if self.transform is not None:
            x = self.transform(x)
        return x, y, metadata

    funcType = types.MethodType
    subset.__getitem__ = funcType(new_fn, subset)
    return subset
