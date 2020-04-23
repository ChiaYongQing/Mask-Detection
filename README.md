## Mask-Detection

### Import Libraries
```
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import scipy.io
import skimage.draw
```
### Setup project directoru
```
ROOT_DIR = os.path.abspath("./Mask_RCNN-master")
DATA_PATH = "./project_1b"
```
### Import Mask RCNN
```
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

%matplotlib inline 

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
```
### Configuration for training on the mask datase
```
class MaskConfig(Config):
    """
    Derived from the base Config class and overrides values specific to the mask dataset.
    """
    NAME = "mask"

    # Train on 1 GPU and 1 images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 2  # background + mask

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    TRAIN_ROIS_PER_IMAGE = 32

    STEPS_PER_EPOCH = 100

    VALIDATION_STEPS = 10
    
    DETECTION_MIN_CONFIDENCE = 0.7
    
config = MaskConfig()
config.display()   
```
```
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax
```
```
class MyMaskDataset(utils.Dataset):
    def load_mask_data_label(self, dataset_dir, subset):
        # Add class
        self.add_class("mask", 1, "mask")
               assert subset in ["train", "val"]
        
        dataset_dir = os.path.join(dataset_dir, subset)

        mat = scipy.io.loadmat(os.path.join(dataset_dir, "labels.mat"))
        if subset == "train":
            annotations = mat['label_train']
            coords_index = 2
            occluder_type_index = 12
            occluder_coords_index_start = 8
            occluder_coords_index_end = 12
        else:
            annotations = mat['LabelTest']
            coords_index = 1
            occluder_type_index = 9
            occluder_coords_index_start = 5
            occluder_coords_index_end = 9
                
        # Add images
        (dir_path, dir_names, filenames_img) = next(os.walk(os.path.abspath(dataset_dir)))
        filenames_img = [i for i in filenames_img if i[-3:] == "jpg"]
        for i in enumerate(filenames_img):
            num = int(i[0])
            name = i[1]   
            image_path = os.path.join(dataset_dir, name)
            img = cv.imread(image_path)
            bb_list = [] # store all bounding boxes coordinates of the image
            not_human_body = True
            # each image may have more than 1 mask
            for j in range(len(annotations[0][num][coords_index])):
                # choose occluder type not a human body
                if int(annotations[0][num][coords_index][j][occluder_type_index]) == 3:
                    not_human_body = False
                else:
                    bb_dict = { "all_points_y": [], "all_points_x": [] }  
                    face = annotations[0][num][coords_index][j][:2]
                    musk = annotations[0][num][coords_index][j][occluder_coords_index_start:occluder_coords_index_end]
                    
                    # load_mask() needs the image size to convert polygons to masks.
                    height, width = img.shape[:2]
                    
                    x_left = face[0] + musk[0]
                    x_right = face[0] + musk[2]
                    y_bottom = face[1] + musk[1]
                    y_top = face[1] + musk[3]

                    # add points (top left, top right, bottom right, bottom left)
                    bb_dict["all_points_x"].append(x_left)
                    bb_dict["all_points_x"].append(x_right)
                    bb_dict["all_points_x"].append(x_right)
                    bb_dict["all_points_x"].append(x_left)

                    bb_dict["all_points_y"].append(y_top)
                    bb_dict["all_points_y"].append(y_top)
                    bb_dict["all_points_y"].append(y_bottom)
                    bb_dict["all_points_y"].append(y_bottom)
                    
                    bb_list.append(bb_dict)
             
                if not_human_body:
                    self.add_image(
                        "mask", 
                        image_id=name, 
                        path=image_path, 
                        width=width, height=height,
                        polygons=bb_list)
                
                
    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a mask dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "mask":
            return super(self.__class__, self).load_mask(image_id)

        # Convert coordinates to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "mask":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
```
### Training dataset
```
dataset_train = MyMaskDataset()
dataset_train.load_mask_data_label(DATA_PATH, "train")
dataset_train.prepare()
```
### Validation dataset
```
dataset_val = MyMaskDataset()
dataset_val.load_mask_data_label(DATA_PATH, "val")
dataset_val.prepare()
```
### Load and display random samples
```
image_ids = np.random.choice(dataset_train.image_ids, 5)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names, limit=1)
```
### Create model in training mode
```
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

init_with = "last"  # imagenet, coco, last, or trained

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    model.load_weights(model.find_last(), by_name=True)
elif init_with == "trained":
    model_path = os.path.join(MODEL_DIR, "mask_faster_rcnn_heads.h5")
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
```
Train in two stages:

1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, pass layers='heads' to the train() function.
2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process. Simply pass layers="all to train all layers.

### Training the head branches
```
# Passing layers="heads" freezes all layers except the head layers. 
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=50, 
            layers='heads')

model_path = os.path.join(MODEL_DIR, "mask_faster_rcnn_heads.h5")
model.keras_model.save_weights(model_path)
```
### Fine tune all layers
```
# Passing layers="all" trains all layers. You can also pass a regular expression to select which layers to train by name pattern.
model.train(dataset_train, dataset_val, 
             learning_rate=config.LEARNING_RATE / 10,
             epochs=20, 
             layers="all")

 model_path = os.path.join(MODEL_DIR, "mask_faster_rcnn_all.h5")
 model.keras_model.save_weights(model_path)
```
```
class InferenceConfig(MaskConfig):
    def __init__(self, min_confidence):
        self.DETECTION_MIN_CONFIDENCE = min_confidence # DETECTION_MIN_CONFIDENCE will be overwritten
        MaskConfig.__init__(self)
```
### Evaluation
```
def evaluate(inference_config, model):
    # Compute VOC-Style mAP @ IoU=0.5
    # Running on 100 random validation images.
    np.random.seed(0)
    image_ids = np.random.choice(dataset_val.image_ids, 200)
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =modellib.load_image_gt(dataset_val, inference_config,
                                       image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                 r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)

    # print(APs)
    print("min detection confidence:", inference_config.DETECTION_MIN_CONFIDENCE,"- mAP: ", np.mean(APs))
```
```
# Find the optimal min detection confidence
model_path = os.path.join(MODEL_DIR, "mask_faster_rcnn_heads.h5")

for confidence_lvl in range(30, 100, 5):
    inference_config = InferenceConfig(confidence_lvl/100)
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)
    # Load trained weights
    model.load_weights(model_path, by_name=True)
    evaluate(inference_config, model)   
```
### Detection
```
# Set the detection confidence to 0.9
inference_config = InferenceConfig(0.9) 

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
model_path = os.path.join(MODEL_DIR, "mask_faster_rcnn_heads.h5")
# model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
```
```
# Display ground truth
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inference_config, 
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                            dataset_train.class_names, figsize=(8, 8))
```
```
# Make detections
results = model.detect([original_image], verbose=1)
r = results[0]

# Display predicted mask
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_val.class_names, r['scores'], ax=get_ax())
model.load_weights(model_path, by_name=True)
```
### Testing on sample images
```
sample_image_path = os.path.join(ROOT_DIR, "datasets/samples")
(dir_path, dir_names, filenames_img) = next(os.walk(os.path.abspath(sample_image_path)))
filenames_img = [i for i in filenames_img if i[-3:] == "jpg"]
samples_images = []
for i in enumerate(filenames_img):
    name = i[1]   
    image_path = os.path.join(sample_image_path, name)
    img = cv.imread(image_path)
    # make detection
    results = model.detect([img], verbose=1)
    r = results[0]
    visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                                dataset_val.class_names, r['scores'], ax=get_ax())
```
