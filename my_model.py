import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

import os
import pathlib
import glob
import time

from six import BytesIO
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util





def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict

def pull_boxes(dict_,category_index):
    # This is the way I'm getting my coordinates
   boxes = dict_['detection_boxes']
# get all boxes from an array
   max_boxes_to_draw = boxes.shape[0]
# get scores to get a threshold
   scores = dict_['detection_scores']
   classId=dict_['detection_classes']
# this is set as a default but feel free to adjust it to your needs
   min_score_thresh=.7
# iterate over all objects found
   boxes_draw=[]
   class_id=[]
   class_score=[]
   class_name=[]


   for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if classId[i]==1 and scores[i] > min_score_thresh:
            names =category_index[dict_['detection_classes'][i]]['name']
            Id = dict_['detection_classes'][i]
            score=dict_['detection_scores'][i]
            dim=dict_['detection_boxes'][i]
            boxes_draw.append(dim)
            class_id.append(Id)
            class_name.append(names)
            class_score.append(score)
   return  boxes_draw, class_name, class_score
       
        
def draw_boxes(boxes, name,score,Image):
    im_width=Image.shape[1]
    im_height=Image.shape[0]
    im = Image.copy()
    for i in range(len(boxes)):
        ymin, xmin, ymax, xmax = list(boxes[i])
    
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, 
                                 ymin * im_height, ymax * im_height)
        
        #im=Image.copy()
        im=cv2.rectangle(im, (int(left), int(top)), (int(right), int(bottom)), (0,255,255), 6)
        im=cv2.putText(im.copy(), name[i]+' '+str("%.2f" % score[i]), (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255,0,0), 3)   
    return im




def processimage(path):
    image = np.array(Image.open(path))
    image_np = np.array(image)
    if image_np.shape[2]<=4:
        image_np=image_np.copy()
        image_np=image_np[:,:,:3]
    
    return image_np

def detect_image_(Image_path,model,category_index):
    Image_=processimage(Image_path)
    output_dict=run_inference_for_single_image(model,Image_)
    boxes, name,score=pull_boxes(output_dict, category_index)
    
    if boxes != []:
        #print(boxes)
        Image_=draw_boxes(boxes,name,score,Image_)
        
    return Image_,boxes,score





