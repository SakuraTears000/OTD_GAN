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
  bash scripts/otd/train_otd_main.sh PATH_TO_DATASET rsicd otd_config MODULE_SAVE_PATH
  ```
  where 'PATH_TO_DATSET' should be replaced with the path to the dataset. 'MODULE_SAVE_PATH' is used to specify the file directory to save the trained OTD_module.
### 2. Train the OTD_GAN
  - Replace 'PATH_TO_DATASET' & 'MODULE_SAVE_PATH' with your own settings in './cfg/rsicd.yml'.
  - Train the OTD-GAN using
  ```
  bash scripts/train.sh ./cfg/rsicd.yml
  ```
### 3. Inference(Sampling)
  - You can sampling images with trained modle using
  - Replace 'PATH_TO_TRAINED_MODEL' & 'SAVE_DIR' with your own settings in 'scripts/test.sh'.
  ```
  bash scripts/test.sh ./cfg/rsicd.yml
  ```
---
### Cite

If you find OTD-GAN useful in your research, please consider citing it:
```

@article{zhao2025remote,
  title={Remote Sensing Image Generation via Object Text Decoupling},
  author={Zhao, Wenda and Zhang, Zhepu and Zhao, Fan and Wang, Haipeng and He, You and Lu, Huchuan},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
  publisher={IEEE}
}

```

**Reference**
- [GALIP: Generative Adversarial CLIPs for Text-to-Image Synthesis] [[code]](https://github.com/tobran/GALIP)
