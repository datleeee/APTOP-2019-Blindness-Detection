# APTOS 2019 Blindness Detection

**Problem:** Detect diabetic retinopathy to stop blindness before it's too late.

**Data:**
* Images 
    * 3662 train
    * 1928 test
    * Severity masks scale [0,1,2,3,4]
    * Noise in both images and labels 
* Training labels, test set:
    * train.csv
    * test.csv (must predict the diagnosis value for these)

**Evaluation:** Quadratic Weight Kappa

## 1st place solution
**Models:** 
* 2 x inception-resnet-v2
* 2 x inception-v4 
* 2 x se-resnext50
* 2 x se-resnext101


**Data:** 2015 data, input size = 512

**Prepocessing**: No special, just resizing

**Augmentation:**
contrast_range=0.2,
brightness_range=20.,
hue_range=10.,
saturation_range=20.,
blur_and_sharpen=True,
rotate_range=180.,
scale_range=0.2,
shear_range=0.2,
shift_range=0.2,
do_mirror=True,

**Loss:** nn.SmoothL1Loss() (simplify the emsembling process)

**Pooling:** 
* Last pooling layer: generalized mean pooling
* [Paper](https://arxiv.org/pdf/1711.02512.pdf)
* [Code](https://github.com/filipradenovic/cnnimageretrieval-pytorch)

**Training:** 2 stages
* 1st stage: train 8 models, evaluate in pairs, stepsize = 5
* 2nd stage: pseudo-labelled on test dataset

## 2nd place solution

**Models:**
* Efficient-net: b3, b4, b5 network
* (se-resnext50, se-resnext101: slow to train, performed poorly)

**Data:** 2015 pretrained data, 2019 trained data

**Preprocessing:** cropped black background and resize

**Augmentation:**
* [albumentations](https://github.com/albumentations-team/albumentations) library
* Blur, Flip, RandomBrightnessContrast, ShiftScaleRotate, ElasticTransform, Transpose, GridDistortion, HueSaturationValue, CLAHE, CoarseDropout

**Training:**
* Pretrain 2015 data: 80 epochs on b3, 15 epochs on b5
* Train current data: 50 epochs on b3, 15 epochs on b5
* Pseudo-labeled (huge improvement): old and current test data --> fine-tune using train+pseudolabeled test

**TTA:** flips

## 4th and 5th place solution

**Models:** EfficientNet
**Training:** 
* pretrain on 2015 dataset for 25 epochs without validation. 
* 5-fold CV on 2019 dataset

## 7th place solution

**Models:** SeResNext50, SeResNext101, InceptionV4
**Preprocessing:** 
**Training:** 
* pretrain 2015 dataset
* pseudo-label test dataset

**TTA:** horizontal flip

**Loss:** Cauchy Loss

**Optimizer:** RAdam (default optimized)

## Approaches work best

**Models:** 
* EfficientNet (best)
* SeResNext50, SeResNext101, InceptionV4

**Augmentation:** [albumentations](https://github.com/albumentations-team/albumentations) library

**Training:**
* pretrain 2015 dataset, train 2019 dataset
* pseudo-labeled on test dataset
* CV 5-fold