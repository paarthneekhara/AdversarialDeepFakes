"""
Create adversarial video that fools xceptionnet.

Usage:
python attack.py
    -i <folder with video files or path to video file>
    -m <path to model file>
    -o <path to output folder, will write one or multiple output videos there>

built upon the code by Andreas Rössler for detecting deep fakes.
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
from torch import autograd
import numpy
from torchvision import transforms

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



def un_preprocess_image(image, size):
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
    

def torch_arctanh(x, eps=1e-6):
    x *= (1. - eps)
    return (torch.log((1 + x) / (1 - x))) * 0.5

def carlini_wagner_attack(input_img, model, cuda = True, 
    max_attack_iter = 500, alpha = 0.005, 
    const = 1e-3, max_bs_iter = 5, confidence = 20.0):
    
    attack_w = autograd.Variable(torch_arctanh(input_img.data - 1), requires_grad = True)
    bestl2 = 1e10
    bestscore = -1

    lower_bound_c = 0
    upper_bound_c = 1.0
    bestl2 = 1e10
    bestimg = None
    optimizer = torch.optim.Adam([attack_w], lr=alpha)
    for bsi in range(max_bs_iter):
        for iter_no in range(max_attack_iter):
            # if output[0][0] - output[0][1] > desired_acc:
            #     # no need to optimize beyond this
            #     break
            

            adv_image = 0.5 * ( torch.tanh(input_img + attack_w) + 1. )
            prediction, output, logits = predict_with_model(adv_image, model, cuda=cuda)
            loss1 = torch.clamp( logits[0][1]-logits[0][0] + confidence, min = 0.0)
            loss2 = torch.norm(adv_image - input_img, 2)

            loss_total = loss2 + const * loss1
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            # print attack_w.grad

            # attack_w.data = attack_w.data - alpha * attack_w.grad

            if iter_no % 50 == 0:
                print ("BSI {} ITER {}".format(bsi, iter_no), output )
                print ("Losses", loss_total, loss1.data, loss2)


        # binary search for c
        if (logits[0][0] - logits[0][1] > confidence):
            if loss2 < bestl2:
                bestl2 = loss2
                print ("best l2", bestl2)
                bestimg = adv_image.detach().data

            upper_bound_c = min(upper_bound_c, const)
        else:
            lower_bound_c = max(lower_bound_c, const)

        const = (lower_bound_c + upper_bound_c)/2.0

    if bestimg is not None:
        return bestimg
    else:
        return adv_image


def iterative_fgsm(input_img, model, cuda = True, max_iter = 100, alpha = 1/255.0, eps = 16/255.0, desired_acc = 0.99):
    input_var = autograd.Variable(input_img, requires_grad=True)

    target_var = autograd.Variable(torch.LongTensor([0]))
    if cuda:
        target_var = target_var.cuda()

    iter_no = 0
    
    while iter_no < max_iter:
        prediction, output, logits = predict_with_model(input_var, model, cuda=cuda)    
        if (output[0][0] - output[0][1]) > desired_acc:
            break
            
        loss_criterion = nn.CrossEntropyLoss()
        loss = loss_criterion(logits, target_var)
        loss.backward()

        step_adv = input_var.detach() - alpha * torch.sign(input_var.grad.detach())
        total_pert = step_adv - input_img
        total_pert = torch.clamp(total_pert, -eps, eps)
        
        input_adv = input_img + total_pert
        input_adv = torch.clamp(input_adv, 0, 1)
        
        input_var.data = input_adv.detach()

        iter_no += 1

    return input_var

def predict_with_model_legacy(image, model, post_function=nn.Softmax(dim=1),
                       cuda=True):
    """
    Predicts the label of an input image. Preprocesses the input image and
    casts it to cuda if required

    :param image: numpy image
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = fake, 0 = real)
    """
    # Preprocess
    preprocessed_image = preprocess_image(image, cuda, legacy = True)

    # Model prediction
    output = model(preprocessed_image)
    output = post_function(output)

    # Cast to desired
    _, prediction = torch.max(output, 1)    # argmax
    prediction = float(prediction.cpu().numpy())

    return int(prediction), output

def predict_with_model(preprocessed_image, model, post_function=nn.Softmax(dim=1), cuda=True):
    """
    Adapted predict_for_model for attack. Differentiable image pre-processing.
    Predicts the label of an input image. Performs resizing and normalization before feeding in image.

    :param image: numpy image
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = fake, 0 = real), output probs, logits
    """
    
    # Model prediction

    # differentiable resizing: doing resizing here instead of preprocessing
    resized_image = nn.functional.interpolate(preprocessed_image, size = (299, 299), mode = "bilinear", align_corners = True)
    norm_transform = xception_default_data_transforms['normalize']
    normalized_image = norm_transform(resized_image)
    
    logits = model(normalized_image)
    output = post_function(logits)

    # Cast to desired
    _, prediction = torch.max(output, 1)    # argmax
    prediction = float(prediction.cpu().numpy())
    # print ("prediction", prediction)
    # print ("output", output)
    return int(prediction), output, logits


def create_adversarial_video(video_path, model_path, output_path,
                            start_frame=0, end_frame=None, attack="iterative_fgsm", cuda=True, showlabel = True):
    """
    Reads a video and evaluates a subset of frames with the a detection network
    that takes in a full frame. Outputs are only given if a face is present
    and the face is highlighted using dlib.
    :param video_path: path to video file
    :param model_path: path to model file (should expect the full sized image)
    :param output_path: path where the output video is stored
    :param start_frame: first frame to evaluate
    :param end_frame: last frame to evaluate
    :param cuda: enable cuda
    :return:
    """
    print('Starting: {}'.format(video_path))

    # Read and write
    reader = cv2.VideoCapture(video_path)

    video_fn = video_path.split('/')[-1].split('.')[0]+'.avi'
    os.makedirs(output_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'HFYU') # Chnaged to HFYU because it is lossless
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None

    # Face detector
    face_detector = dlib.get_frontal_face_detector()

    # Load model
    model, *_ = model_selection(modelname='xception', num_out_classes=2)
    if model_path is not None:
        if not cuda:
            model = torch.load(model_path, map_location = "cpu")
        else:
            model = torch.load(model_path)
        print('Model found in {}'.format(model_path))
    else:
        print('No model found, initializing random model.')
    if cuda:
        print("Converting mode to cuda")
        model = model.cuda()
        for param in model.parameters():
            param.requires_grad = True
        print("Converted to cuda")

    # Text variables
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
        if writer is None:
            writer = cv2.VideoWriter(join(output_path, video_fn), fourcc, fps,
                                     (height, width)[::-1])

            # writer = cv2.VideoWriter(join(output_path, video_fn), 0, 1,
            #                          (height, width)[::-1])

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
            
            # Attack happening here
            if attack == "iterative_fgsm":
                perturbed_image = iterative_fgsm(processed_image, model, cuda)
            elif attack == "carlini_wagner":
                perturbed_image = carlini_wagner_attack(processed_image, model, cuda)
            
            # Undo the processing of xceptionnet
            unpreprocessed_image = un_preprocess_image(perturbed_image, size)
            image[y:y+size, x:x+size] = unpreprocessed_image
            

            cropped_face = image[y:y+size, x:x+size]
            processed_image = preprocess_image(cropped_face, cuda = cuda)
            prediction, output, logits = predict_with_model(processed_image, model, cuda=cuda)

            print (">>>>Prediction for frame no. {}: {}".format(frame_num ,output))

            prediction, output = predict_with_model_legacy(cropped_face, model, cuda=cuda)

            print (">>>>Prediction LEGACY for frame no. {}: {}".format(frame_num ,output))

            if showlabel:
                # Text and bb
                # print a bounding box in the generated video
                x = face.left()
                y = face.top()
                w = face.right() - x
                h = face.bottom() - y
                label = 'fake' if prediction == 1 else 'real'
                color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
                output_list = ['{0:.2f}'.format(float(x)) for x in
                               output.detach().cpu().numpy()[0]]

                cv2.putText(image, str(output_list)+'=>'+label, (x, y+h+30),
                            font_face, font_scale,
                            color, thickness, 2)
                # draw box over face
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        if frame_num >= end_frame:
            break

        # Show
        #cv2.imshow('test', image)
        #cv2.waitKey(33)     # About 30 fps
        writer.write(image)
    pbar.close()
    if writer is not None:
        writer.release()
        print('Finished! Output saved under {}'.format(output_path))
    else:
        print('Input video file was empty')


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video_path', '-i', type=str)
    p.add_argument('--model_path', '-mi', type=str, default=None)
    p.add_argument('--output_path', '-o', type=str,
                   default='.')
    p.add_argument('--start_frame', type=int, default=0)
    p.add_argument('--end_frame', type=int, default=None)
    p.add_argument('--attack', '-a', type=str, default="iterative_fgsm")
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--showlabel', action='store_true') # add face labels in the generated video

    args = p.parse_args()

    video_path = args.video_path
    if video_path.endswith('.mp4') or video_path.endswith('.avi'):
        create_adversarial_video(**vars(args))
    else:
        videos = os.listdir(video_path)
        for video in videos:
            args.video_path = join(video_path, video)
            create_adversarial_video(**vars(args))