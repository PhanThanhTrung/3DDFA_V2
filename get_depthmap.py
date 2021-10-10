# coding: utf-8

__author__ = 'cleardusk'

import argparse
import imageio
from tqdm import tqdm
import yaml
import cv2
import os
import pickle
import zlib
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
import numpy as np
# from utils.render_ctypes import render
from utils.functions import cv_draw_landmark, get_suffix
import glob
from utils.depth import depth
def decompress_data(filepath="test.gz"):
    with open(filepath, 'rb') as fp:
        data = zlib.decompress(fp.read())
        successDict = pickle.loads(data)
    return successDict

def get_the_biggest_face(bboxes):
    if len(bboxes)==0:
        return []
    sorted_bboxes = sorted(bboxes, key = lambda bbox: (bbox[2]-bbox[0])*(bbox[3]-bbox[1]), reverse=True)
    return [sorted_bboxes[0]]

if __name__ == '__main__':
    cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)

    gpu_mode = True
    tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
    face_boxes = FaceBoxes()
    
    all_image_list = glob.glob('/root/CelebA/*/*valid/*.jpg')
    for image_path in tqdm(all_image_list[:10]):
        bgr_image = cv2.imread(image_path)
        image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        output_label_path = image_path.replace('/CelebA/','/CelebA_annotation/')[:-3]+'gz'
        if not os.path.exists(output_label_path):
            continue
        label_data = decompress_data(output_label_path)
        if label_data is None:
            continue
        rotation_angle = label_data['rotate_angle']
        boxes = [label_data['bbox'] + [label_data['confidence']]] 
        
        # boxes = face_boxes(image)
        # if len(boxes) ==0:
        #     continue
        # boxes = get_the_biggest_face(boxes)
        param_lst, roi_box_lst = tddfa(image, boxes)

        dense_flag = True
        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
        mask = np.zeros_like(image)
        depth_map = depth(bgr_image, ver_lst, tddfa.tri, show_flag=False)
        
        if rotation_angle == 270:
            depth_map = cv2.rotate(depth_map, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle == 90:
            depth_map = cv2.rotate(depth_map, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation_angle == 180:
            depth_map = cv2.rotate(depth_map, cv2.ROTATE_180)
        else:
            pass

        output_image_path = image_path.replace('/CelebA/','/CelebA_depthmap/')
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        cv2.imwrite(output_image_path, depth_map)