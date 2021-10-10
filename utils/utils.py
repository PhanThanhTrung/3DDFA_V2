import logging
import tempfile
import traceback
from io import BytesIO

import cv2
import fitz
import numpy as np
import unidecode
from decord import VideoReader
from imutils.video import FPS, FileVideoStream
from skimage import transform as trans
import torch
import gc


def pix2np(pix):
    im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.h, pix.w, pix.n)
    im = np.ascontiguousarray(im[..., [2, 1, 0]])  # rgb to bgr
    return im
    
def pdf_to_list_ndarray(filename, data_bytes):
    doc = fitz.Document(stream=data_bytes, filetype='pdf')
    # doc = fitz.open(pdf_path)
    scale= 3.0
    mat = fitz.Matrix(scale, scale)

    output = []
    for index, page in enumerate(doc):
        pix = page.getPixmap(alpha=False, matrix=mat)
        image_page = pix2np(pix)

        output.append(image_page)

    return output

def bytes_video_to_ndarray_imutils(input: bytes = None, fps=5, max_process_frame=-1, get_motion_info=False): 
    if input is None: 
        return []

    output = []
    with tempfile.NamedTemporaryFile() as tfile:
        tfile.write(input)
        
        fvs = FileVideoStream(tfile.name).start()
        # start the FPS timer
        # fps = FPS().start()

        # loop over frames from the video file stream
        while fvs.more():
            # grab the frame from the threaded video file stream, resize
            # it, and convert it to grayscale (while still retaining 3
            # channels)
            frame = fvs.read()
            if frame is not None:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                output.append(frame)
            # fps.update()

        # stop the timer and display FPS information
        # fps.stop()
    if get_motion_info:
        if len(output) <= 1:
            motion_info = None
        else:
            motion_info = get_motion_optical_flow(output)
    else:
        motion_info = None

    if get_motion_info:
        if max_process_frame != -1:
            if len(output) < max_process_frame: 
                return output, motion_info
                
            return [output[i] for i in np.round(np.linspace(0, len(output) - 1, max_process_frame)).astype(int)], motion_info
        else:
            return output, motion_info
        
    else:
        if max_process_frame != -1:
            if len(output) < max_process_frame: 
                return output
                
            return [output[i] for i in np.round(np.linspace(0, len(output) - 1, max_process_frame)).astype(int)]
        else:
            return output


def load_video_from_storage(input, fps=5, max_process_frame=15, max_iter = 500, get_motion_info=False): 
    """
    Load video bytes and ndarray from flask storage
    """
    bytes_data = None
    if isinstance(input, bytes): 
        bytes_data = input
        file_name = "input.mp4"
        file_ext = "mp4"
    elif type(input).__name__ != 'FileStorage': 
        raise MlChainError(
                    msg="Input video can not be None",
                    code="E812",
                    status_code=200
                )

    if bytes_data is None:
        file_name = input.filename
        file_ext = unidecode.unidecode(file_name.split(".")[-1].lower())

    try:
        if bytes_data is None:
            bytes_data = input.read()
        if get_motion_info:
            output_images, motion_info = bytes_video_to_ndarray_imutils(bytes_data, fps=fps, max_process_frame=max_process_frame, get_motion_info=get_motion_info)
        else:
            output_images = bytes_video_to_ndarray_imutils(bytes_data, fps=fps, max_process_frame=max_process_frame, get_motion_info=get_motion_info)
    except Exception as ex: 
        logging.info("READ VIDEO ERROR: {0}".format(traceback.format_exc()))
        raise MlChainError(
            msg="The input video is not in true format of {0}".format(file_ext),
            code="E814",
            status_code=200
        )

    if get_motion_info:
        return file_name, file_ext, bytes_data, output_images, motion_info
    else:
        return file_name, file_ext, bytes_data, output_images

def rotate_90(image: np.ndarray):
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

def rotate_180(image: np.ndarray):
    return cv2.rotate(image, cv2.ROTATE_180)

def rotate_270(image: np.ndarray):
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

def do_rotate_90(image: np.ndarray, times:int=1):
    times = times % 4
    if times == 0:
        return image 
    elif times == 1:
        return rotate_90(image)
    elif times == 2:
        return rotate_180(image)
    elif times == 3:
        return rotate_270(image)
    return None

def align_face_by_landmarks(cv_img, dst, dst_w, dst_h):
    if dst_w == 96 and dst_h == 112:
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041] ], dtype=np.float32)
    elif dst_w == 112 and dst_h == 112:
        src = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041] ], dtype=np.float32)
    elif dst_w == 150 and dst_h == 150:
        src = np.array([
            [51.287415, 69.23612],
            [98.48009, 68.97509],
            [75.03375, 96.075806],
            [55.646385, 123.7038],
            [94.72754, 123.48763]], dtype=np.float32)
    elif dst_w == 160 and dst_h == 160:
        src = np.array([
            [54.706573, 73.85186],
            [105.045425, 73.573425],
            [80.036, 102.48086],
            [59.356144, 131.95071],
            [101.04271, 131.72014]], dtype=np.float32)
    elif dst_w == 224 and dst_h == 224:
        src = np.array([
            [76.589195, 103.3926],
            [147.0636, 103.0028],
            [112.0504, 143.4732],
            [83.098595, 184.731],
            [141.4598, 184.4082]], dtype=np.float32)
    else:
        return None
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]

    if M is None:
        img = cv_img.copy()
        
        #use center crop
        det = np.zeros(4, dtype=np.int32)
        det[0] = int(img.shape[1]*0.0625)
        det[1] = int(img.shape[0]*0.0625)
        det[2] = img.shape[1] - det[0]
        det[3] = img.shape[0] - det[1]

        margin = 44
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
        bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
        ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
        ret = cv2.resize(ret, (dst_w, dst_h))
        return ret 
        
    face_img = cv2.warpAffine(cv_img,M,(dst_w,dst_h), borderValue = 0.0)
    return face_img

def do_flip(data):
    for idx in range(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])


def resize_image(image: np.ndarray, target_size=640):
    height, width = image.shape[:2]
    ratio = 1
    if max(height, width) > target_size:
        ratio = target_size / max(height, width)

        target_h, target_w = int(height * ratio), int(width * ratio)
        image = cv2.resize(image, (target_w, target_h),
                           interpolation=cv2.INTER_CUBIC)
    return image, ratio

def check_face_image_brightness(image, bbox, r_padding=0.2, threshold=0.3):
    '''
        Check image brightness
        Return:
            True:  brightness of face in image is good
            False: brightness of face in image is low
    '''
    def isbright(image, dim=10, thresh=0.3):
        # Resize image to 10x10
        image = cv2.resize(image, (dim, dim), cv2.INTER_CUBIC)
        # Convert color space to LAB format and extract L channel
        L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
        # Normalize L channel by dividing all pixel values with maximum pixel value
        L = L/np.max(L)
        # Return True if mean is greater than thresh else False
        return np.mean(L) > thresh
    
    wid, hei = image.shape[:2]
    x1, y1, x2, y2 = bbox[:4]
    w_f = x2 - x1
    h_f = y2 - y1

    pad_left = pad_right = r_padding * w_f
    pad_top = pad_bot = r_padding * h_f

    x1 = int(max(0, x1-pad_left))
    x2 = int(min(wid, x2+pad_right))
    y1 = int(max(0, y1-pad_top))
    y2 = int(min(hei, y2+pad_bot))
    
    face_image = image[y1:y2, x1:x2, :]
    quality_inp = isbright(face_image, dim=448, thresh=threshold)
    return quality_inp
    
def check_cut_face(image: np.ndarray, landmark):
    """
        Check if face is cutting with image boundary or not.
        Input:
            image_list: List[np.ndarray], a list of bgr image to check
            facial_landmark: List[List[int]]
        ouput:
            A list of boolean value with the same length as image_list, each value coresponding to each image in image_list.
            If an element in list is True mean the coresponding image is cut face.
    """
    
    img_h, img_w = image.shape[:2]

    landmark = np.array(landmark)
    all_x_coor = landmark[..., 0]
    all_y_coor = landmark[..., 1]
    if np.max(all_x_coor) > img_w or \
            np.max(all_y_coor) > img_h or \
            np.min(all_x_coor) < 0 or \
            np.min(all_y_coor) < 0:
        return True
    else:
        return False

def check_small_face_in_image(image, bbox, min_face_size=200):
    """
        Return:
            True: if face is too small.
            False: if face is bigger then predefined threshold.
    """
    image_height, image_width, _ = image.shape
    face_height, face_width = bbox[3] - bbox[1], bbox[2] - bbox[0]
    if min(face_height, face_width) < min_face_size:
        return True
    else:
        return False

def check_two_face_in_image(image, bbox_info):
    """
        Return:
            True: there are more than one faces in the image.
            False: if there is one face in the image.
    """
    image_height, image_width, _ = image.shape
    bbox_list, point_list = bbox_info
    if len(bbox_list) == 1:
        return False

    else: # len(bbox_list) > 1
        list_face_area = []
        for bbox in bbox_list:
            x1, y1, x2, y2 = bbox[:4]
            area_face = (x2-x1) * (y2-y1)
            if list_face_area:
                for old_area in list_face_area:
                    if area_face > 0.7*old_area and old_area > 0.7*area_face:
                        return True
    return False


def get_head_pose(landmarks):
    ''' Check headpose from model landmark 3D
    '''
    center_point = landmarks[30]
    top_left_point = landmarks[0]
    bot_left_point = landmarks[6]
    top_right_point = landmarks[16]
    bot_right_point = landmarks[10]

    if center_point[0] < top_left_point[0]:
        head_pose = 'left'
    elif center_point[0] > top_right_point[0]:
        head_pose = 'right'
    else:
        head_pose = 'center'
    return head_pose

def get_motion_optical_flow(video, ratio_shape=10):
    lk_params = dict(winSize = (30, 30), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    all_gradients = []
    all_norm_gradients = []
    all_angles = []
    if len(video) == 0:
        return {
            'norm_gradient':[],
            'angle': []
        }

    for index, frame in enumerate(video):
        if index == 0:
            first_frame = frame
            prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            hei, wid = prev_gray.shape[:2]

            check_points = []
            for i in range(ratio_shape):
                for j in range(ratio_shape):
                    crd_pnt = [i*wid/ratio_shape + wid/(2*ratio_shape), j*hei/ratio_shape + hei/(2*ratio_shape)]
                    check_points.append(crd_pnt)

            check_points = np.array(check_points)
            check_points = np.expand_dims(check_points, axis=1).astype(np.float32)
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pred_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, check_points, None, **lk_params)

        gradient = pred_points - check_points
        gradient = gradient.squeeze()
        norm_gradient = np.linalg.norm(gradient, axis=1)/np.linalg.norm([wid, hei])
        
        x_gradient = gradient[..., 0]
        y_gradient = gradient[..., 1]

        angle = np.arctan2(y_gradient, x_gradient)
        
        all_gradients.append(gradient)
        all_norm_gradients.append(norm_gradient)
        all_angles.append(angle)
        prev_gray = gray.copy()

    motion_info = {
        "norm_gradient": all_norm_gradients,
        "angle": all_angles
    }

    return motion_info


def empty_cache_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()