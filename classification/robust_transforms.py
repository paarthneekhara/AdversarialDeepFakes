import torch
import torch.nn as nn
from torch import autograd
import numpy
from torchvision import transforms



    

def compress_decompress(image,factor,cuda=True):
  # :param image: tensor image
  # :param cuda: enables cuda, must be the same parameter as the model
  # using interpolate(input, size, scale_factor, mode, align_corners) to upsample
  # uses either size or scale factor

  image_size = list(image.size())
  image_size = image_size[2:]
  compressed_image = nn.functional.interpolate(image, scale_factor = factor, mode = "bilinear", align_corners = True)
  decompressed_image = nn.functional.interpolate(compressed_image, size = image_size, mode = "bilinear", align_corners = True)

  return decompressed_image



def add_gaussian_noise(image,amount,cuda=True):
  # :param image: tensor image
  # :param amount: amount of noise to be added
  # :param cuda: enables cuda, must be the same parameter as the model


  new_image = image + amount*torch.randn_like(image)
  new_image = torch.clamp(new_image, min=0, max=1)

  return new_image


def gaussian_blur(image,kernel_size=(11,11),sigma=(10.5, 10.5), cuda=True):
  # smooths the given tensor with a gaussian kernel by convolving it to each channel. 
  # It suports batched operation.
  # :param image: tensor image
  # :param kernel_size (Tuple[int, int]): the size of the kernel
  # :param sigma (Tuple[float, float]): the standard deviation of the kernel

  gauss = tgm.image.GaussianBlur(kernel_size, sigma)

  # blur the image
  img_blur = gauss(image)

  # convert back to numpy. Turned off because I'm returning tensor
  #image_blur = tgm.tensor_to_image(img_blur.byte())

  return img_blur