import cv2
import numpy as np
import sys
from segment_anything import sam_model_registry, SamPredictor
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch
from MainModel import LightningNN
    
model = LightningNN()
model.load_state_dict(torch.load("MODELP.pt"))
model.eval()

class ThresholdTransform(object):
  def __init__(self, thr_255):
    self.thr = thr_255 / 255.  # input threshold for [0..255] gray level, convert to [0..1]

  def __call__(self, x):
    return (x > self.thr).to(x.dtype)  # do not change the data type

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
    ThresholdTransform(thr_255=147)
])

# def Crop(image, coords: list):
#     # sys.path.append('..')
#     sam_checkpoint = 'sam_vit_h_4b8939.pth'
#     model_type = 'vit_h'
#     device = 'cpu'
#     sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#     sam.to(device=device)
#     predictor = SamPredictor(sam)
    
#     predictor.set_image(image)
    
#     input_coords = np.array([coords])
#     input_label = np.array([1])

#     masks, _, _ = predictor.predict(
#         point_coords=input_coords,
#         point_labels=input_label,
#         multimask_output=False
#     )

    # return masks

def Predict(image):
    # mask = masks[0]
    # color = np.array([255, 255, 255])
    # h, w = mask.shape[-2:]
    # mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    im = Image.fromarray(np.array(image).astype(np.uint8))
    cv2.imwrite('test.png', image)
    test = transform(im)
    
    pred = model(test.reshape(1, 1, 128, 128))
    return torch.argmax(pred), pred, im

def RawPredict(image):
    # im = Image.fromarray(np.array(image).astype(np.uint8))
    cv2.imwrite('image.png', image)
    im = Image.open('image.png')
    im_t = transform(im)
    im_t = transforms.functional.invert(im_t)

    pred = model.model(im_t.reshape(1, 1, 128, 128))
    return torch.argmax(pred)

