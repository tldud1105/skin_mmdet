# Skin Object Detection

---

This repo contains the supported code and configuration files to reproduce skin object detection results based on [mmdetection](https://mmdetection.readthedocs.io/en/latest/)

## Updates

---

***20/08/2023*** Initial commits

## Results and Models

---

### Mask R-CNN

| Number of data | Augmentation | Backbone | Lr Schd | #params | FLOPs | Box mAP |
|:--------------:|:------------:|:--------:|:-------:|:-------:|:-----:|:-------:|
|     1,565      |     None     |  Swin-S  |   1x    |   00M   | 000G  |  00.0   |


## Dataset preparation

---

Currently, only COCO format is supported for annotations. 
Therefore, existing annotation files need to be converted to COCO format. 

You can use the following command to convert Pascal VOC format to COCO format.

### Convert Pascal VOC format to COCO format

**⚠️NOTE⚠️** 
First, you need to define the LABELS variable inside the convert_voc_to_coco.py file as follows.
```
LABELS = {'papule acne': 1,'pustule acne': 2,'nodule acne': 3}
```
#### Command-line
```
# [-h]
python tools/convert_voc_to_coco.py --voc_dir <VOC_DIR> --coco_dir <COCO_DIR>

# [--example]
python tools/convert_voc_to_coco.py --voc_dir ../data/raw/labels(xml) --coco_dir ../data/raw/labels(coco)
```
#### Data Structure
```
├── data
│   └── raw
│       ├── images
│       │   ├── 01na00ej000001kr.jpg
│       │   ├── 01na00ej000002kr.jpg
│       ├── labels(xml)
│       │   ├── 01na00ej000001kr.xml   # Pascal VOC format XML file.
│       │   ├── 01na00ej000002kr.xml
│       ├── labels(coco)
│       │   ├── output.json   # The resulting COCO format JSON file.
```
### Split Dataset

The code in tools/split_coco.py performs the task of splitting a multi-label coco annotation file with preserving class distributions among train and test sets.
It is referenced from [akarazniewicz/cocosplit](https://github.com/akarazniewicz/cocosplit) repo.

```
Pre-installation:
  pip install fancy
  pip install scikit-multilearn
```

```
positional arguments:
  coco_annotations      Path to COCO annotations file.
  train                 Where to store COCO training annotations
  test                  Where to store COCO test annotations
  
optional arguments:
  -s SPLIT              A percentage of a split; a number in (0, 1)
  --having-annotations  Ignore all images without annotations. Keep only these
                        with at least one annotation
  --multi-class         Split a multi-class dataset while preserving class
                        distributions in train and test sets
```

```
# [-h]
python tools/split_coco.py [--having-annotations] [--multi-class] -s SPLIT coco_annotations train test

# [--example]
python tools/split_coco.py --having-annotations --multi-class -s 0.8 ../data/raw/labels(coco)/output.json ../data/raw/labels(coco)/train.json ../data/raw/labels(coco)/test.json
```

## Installation

---

### 0. Create a conda environment
```
conda create --name skin_mmdet python=3.8 -y
activate skin_mmdet
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 1. Install MMEngine and MMCV using MIM.

Step 0. Install MMEngine and MMCV using MIM.
```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```
Step 1. Install MMDetection.
```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```

If you want more detailed installation instructions, please refer to [the following link](https://mmdetection.readthedocs.io/en/latest/get_started.html).

## Training

---

### Case 0. <span style="background-color: #fff5b1">Mask R-CNN</span> model, <span style="background-color: #dcffe4">Swin Transformer</span> backbone

#### Prepare config files
- Add skin_mmdet\mmdetection\configs\swin\mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_skin.py
- Add skin_mmdet\mmdetection\configs\swin\mask-rcnn_swin-t-p4-w7_fpn_amp-ms-crop-3x_skin.py
- Add skin_mmdet\mmdetection\configs\_base_\datasets\skin_detection.py
- Add skin_mmdet\mmdetection\configs\_base_\models\mask-rcnn_r50_fpn_skin.py

#### Register the dataset in mmdet.registry.
- Add skin_mmdet\mmdetection\mmdet\datasets\skin.py
- Add 'SkinDataset' in skin_mmdet\mmdetection\mmdet\datasets\__init__.py

#### Run
```
[-h]
cd mmdetection
python tools/train.py <CONFIG_FILE>

[--example]
cd mmdetection
python tools/train.py configs/swin/mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_skin.py
```



