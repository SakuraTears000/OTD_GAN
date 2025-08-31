#  Remote Sensing Image Generation via Object Text Decoupling
  Implementation of "Remote Sensing Image Generation via Object Text Decoupling" in Pytorch

## Requirements
  - Python == 3.9
  - Pytorch >= 1.9.0
  - At least one NVIDIA GPU with more than 24 GB VRAM

## Installation
Clone this repo.
```
git clone https://github.com/SakuraTears000/OTD_GAN
pip install -r requirements.txt
```
Install [CLIP](https://github.com/openai/CLIP)

## Preparation
### Datasets
  We conduct experiments on two publicly available remote sensing datasets: NWPU-Captions and RSICD.

  - NWPU-Captions: 7700 images from 11 categories of RS scenarios with resolution of 256*256. 6160 images are used for training, with each image corresponding to 5 textual captions, 1540 images are employed for evaluation.
  - RSICD:  2306 images with resolution of 224*224 from 6 RS categories. 1842 text-image pairs are used for training, with each image corresponding to 5 textual captions, and the rest 464 pairs for evaluation.

  You can download the preprocessed datasets from [Baidu Netdisk](https://pan.baidu.com/s/1xQFNwlIa_cIKEQIyoIiHZQ?pwd=bc6e), then extract them to `dataset/`.

## Training (Taking RSICD dataset as example)
### 1.Train the OTD module
  - Clone and install [CoCoOp](https://github.com/KaiyangZhou/CoOp), then move '/code/train_otd' to the folder.
  - Train the OTD module using
  ```
  bash scripts/otd/train_otd_main.sh rsicd otd_config
  ```
 

