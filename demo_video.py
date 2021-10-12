# coding: utf-8

__author__ = 'cleardusk'

import argparse
import imageio
from tqdm import tqdm
import yaml
import glob
import zlib
import pickle
from pathlib import Path
import os
import cv2
import tempfile
import numpy as np
from imutils.video import FPS, FileVideoStream
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
from utils.depth import depth
from utils.utils import bytes_video_to_ndarray_imutils
from utils.functions import cv_draw_landmark, get_suffix


def compress_data(data, filepath="test.gz"):
    with open(filepath, 'wb') as f:
        f.write(zlib.compress(pickle.dumps(data, pickle.HIGHEST_PROTOCOL), 9))


def decompress_data(filepath="test.gz"):
    with open(filepath, 'rb') as fp:
        data = zlib.decompress(fp.read())
        successDict = pickle.loads(data)
    return successDict

def do_crop_face(image, bbox, padding_ratio):
    if isinstance(padding_ratio, list):
        padding_width, padding_height = padding_ratio
        height, width = image.shape[:2]
        left, top, right, bottom = bbox[:4]
        face_height, face_width = bottom - top, right - left

        padding = [
            0 - face_width*padding_width,
            0 - face_height*padding_height,
            face_width*padding_width,
            face_height*padding_height
        ]
    else:
        max_padding_ratio, min_padding_ratio = padding_ratio + 0.1, padding_ratio - 0.1
        height, width = image.shape[:2]
        left, top, right, bottom = bbox[:4]
        face_height, face_width = bottom - top, right - left

        padding = [
            0 - face_width*random.uniform(min_padding_ratio, max_padding_ratio),
            0 - face_height*random.uniform(min_padding_ratio, max_padding_ratio),
            face_width*random.uniform(min_padding_ratio, max_padding_ratio),
            face_height*random.uniform(min_padding_ratio, max_padding_ratio)
        ]

    coord = np.array(bbox[:4])
    padding = np.array(padding)
    new_coord = coord + padding
    new_left = int(max(new_coord[0], 0))
    new_top = int(max(new_coord[1], 0))
    new_right = int(min(new_coord[2], width))
    new_bottom = int(min(new_coord[3], height))

    cropped_img = image[new_top: new_bottom, new_left: new_right, :]
    return cropped_img

def most_frequent(List):
    return max(set(List), key = List.count)

if __name__ == '__main__':
    cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
    gpu_mode = True
    tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
    face_boxes = FaceBoxes()

    all_video_path = glob.glob('/root/valid___not_classify_review_2707/*/*/*.mp4') + glob.glob('/root/valid___not_classify_review_2707/*/*/*.mov')
    need_to_review = []
    for video_path in tqdm(all_video_path):
        base_name_video = os.path.basename(video_path)
        path = Path(video_path)
        with open(path, 'rb') as f:
            video=f.read()
        output_label_path = video_path.replace('/valid___not_classify_review_2707/','/valid___not_classify_review_2707_annotation/')[:-3]+'gz'
        if not os.path.exists(output_label_path):
            continue
        label_data = decompress_data(output_label_path)
        if label_data is None:
            continue
        all_frame, motion_info = bytes_video_to_ndarray_imutils(
            video, max_process_frame=-1, get_motion_info=True)

        frame_shape = [frame.shape[:2] for frame in all_frame]
        if len(set(frame_shape)) >=1:
            need_to_review.append(video_path)
        frame_height, frame_width = most_frequent(frame_shape)

        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        output_path =video_path.replace('/valid___not_classify_review_2707/','/valid___not_classify_review_2707_depthmap/')
        if os.path.exists(output_path):
            continue
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))


        rotation_angle = label_data['rotate_angle']
        frame_info = label_data['frame_info']
        if frame_info is None:
            continue
        count = 0

        # run
        for index, frame in enumerate(all_frame):
            if rotation_angle == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif rotation_angle == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation_angle == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            else:
                pass

            depth_map = np.zeros_like(frame)
            if frame_info[index] is not None:
                boxes = [frame_info[index]['bbox'] + [0.99]] 

                param_lst, roi_box_lst = tddfa(frame, boxes)

                ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
                depth_map = depth(depth_map, ver_lst, tddfa.tri, show_flag=False)

            if rotation_angle == 270:
                frame = cv2.rotate(depth_map, cv2.ROTATE_90_CLOCKWISE)
            elif rotation_angle == 90:
                frame = cv2.rotate(depth_map, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif rotation_angle == 180:
                frame = cv2.rotate(depth_map, cv2.ROTATE_180)
            else:
                pass

            writer.write(depth_map[..., ::-1])

        video.release()
        writer.release()

    with open('/root/review.txt','w') as f:
        for line in need_to_review:
            f.write(line+'\n')


