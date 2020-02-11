"""
Evaluates a folder of video files or a single file with a xception binary
classification network.

Usage:
python detect_from_video.py
    -i <folder with video files or path to video file>
    -m <path to model file>
    -o <path to output folder, will write one or multiple output videos there>

Author: Andreas Rössler
"""
import os
import argparse
from os.path import join
import cv2
import dlib
import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm import tqdm

from network.models import model_selection
from dataset.transform import xception_default_data_transforms
import robust_transforms
from torchvision import transforms
import numpy

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def preprocess_image(image, cuda=True, legacy = False):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.

    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    # Revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    if not legacy:
        # only conver to tensor here, 
        # other transforms -> resize, normalize differentiable done in predict_from_model func
        preprocess = xception_default_data_transforms['to_tensor']
    else:
        preprocess = xception_default_data_transforms['test']

    preprocessed_image = preprocess(pil_image.fromarray(image))
    
    # Add first dimension as the network expects a batch
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()

    preprocessed_image.requires_grad = True
    return preprocessed_image


def un_preprocess_image(image):
    """
    Tensor to PIL image and RGB to BGR
    """
    
    image.detach()
    new_image = image.squeeze(0)
    new_image = new_image.detach().cpu()

    undo_transform = transforms.Compose([
        transforms.ToPILImage(),
    ])

    new_image = undo_transform(new_image)
    new_image = numpy.array(new_image)

    new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)

    return new_image

def test_transforms(video_path, output_path, cuda=True, start_frame = 0, end_frame=None):
    print('Starting: {}'.format(video_path))

    # Read and write
    reader = cv2.VideoCapture(video_path)

    video_fn = video_path.split('/')[-1].split('.')[0]+'.avi'
    os.makedirs(output_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None

    # Face detector
    face_detector = dlib.get_frontal_face_detector()

    # image transform
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1

    # Frame numbers and length of output video
    frame_num = 0
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    pbar = tqdm(total=end_frame-start_frame)

    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break
        frame_num += 1

        if frame_num < start_frame:
            continue
        pbar.update(1)

        # Image size
        height, width = image.shape[:2]

        # Init output writer
        

        # 2. Detect with dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            # For now only take biggest face
            face = faces[0]

            # --- Prediction ---------------------------------------------------
            # Face crop with dlib and bounding box scale enlargement
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y+size, x:x+size]

            processed_image = preprocess_image(cropped_face, cuda = cuda)

            transformed_image = robust_transforms.compress_decompress(processed_image,factor=0.2,cuda=cuda)


            original_image = un_preprocess_image(processed_image)
            new_image = un_preprocess_image(transformed_image)

            cv2.imwrite(join(output_path, "original.jpg"), original_image)
            cv2.imwrite(join(output_path, "transformed.jpg"), new_image)

            
            
            break
        break
        
    


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video_path', '-i', type=str)
    p.add_argument('--output_path', '-o', type=str)
    p.add_argument('--cuda', action='store_true')
    args = p.parse_args()

    video_path = args.video_path
    if video_path.endswith('.mp4') or video_path.endswith('.avi'):
        test_transforms(**vars(args))
    else:
        videos = os.listdir(video_path)
        for video in videos:
            args.video_path = join(video_path, video)
            test_transforms(**vars(args))
