import os, sys
import os.path as osp
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import clip
import importlib
from lib.utils import choose_model


###########   preparation   ############
def load_clip(clip_info, device):
    import clip as clip
    model = clip.load(clip_info['type'], device=device)[0]
    return model


def prepare_models(args):
    device = args.device
    local_rank = args.local_rank
    multi_gpus = args.multi_gpus
    CLIP4trn = load_clip(args.clip4trn, device).eval()
    CLIP4evl = load_clip(args.clip4evl, device).eval()
    NetG,NetD,NetC,CLIP_IMG_ENCODER,CLIP_TXT_ENCODER,CLIP_TXT_ENCODER_WITH_OBJ,CLIP_OBJ_ENCODER,PromptLearner= choose_model(args.model)
    # image encoder
    CLIP_img_enc = CLIP_IMG_ENCODER(CLIP4trn).to(device)
    for p in CLIP_img_enc.parameters():
        p.requires_grad = False
    CLIP_img_enc.eval()
    # text encoder
    CLIP_txt_enc = CLIP_TXT_ENCODER(CLIP4trn).to(device)
    for p in CLIP_txt_enc.parameters():
        p.requires_grad = False
    CLIP_txt_enc.eval()

    CLIP_txt_enc_obj = CLIP_TXT_ENCODER_WITH_OBJ(CLIP4trn, args.n_ctx, args.cname2lab).to(device)
    for p in CLIP_txt_enc_obj.parameters():
        p.requires_grad = False
    CLIP_txt_enc_obj.eval()

    CLIP_obj_enc = CLIP_OBJ_ENCODER(CLIP4trn).to(device)
    for p in CLIP_obj_enc.parameters():
        p.requires_grad = False
    CLIP_obj_enc.eval()

    PL = PromptLearner(args.n_cls, args.n_ctx, args.cond_dim).to(device)

    # GAN models
    netG = NetG(args.nf, args.z_dim, args.cond_dim, args.imsize, args.ch_size, args.mixed_precision, CLIP4trn).to(device)
    netD = NetD(args.nf, args.imsize, args.ch_size, args.mixed_precision).to(device)
    netC = NetC(args.nf, args.cond_dim, args.mixed_precision).to(device)
    if (args.multi_gpus) and (args.train):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        netG = torch.nn.parallel.DistributedDataParallel(netG, broadcast_buffers=False,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank, find_unused_parameters=True)
        netD = torch.nn.parallel.DistributedDataParallel(netD, broadcast_buffers=False,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank, find_unused_parameters=True)
        netC = torch.nn.parallel.DistributedDataParallel(netC, broadcast_buffers=False,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank, find_unused_parameters=True)
    return CLIP4trn, CLIP4evl, CLIP_img_enc, CLIP_txt_enc, CLIP_txt_enc_obj, CLIP_obj_enc, netG, netD, netC, PL


def prepare_dataset(args, split, img_transform, obj_transform):
    if args.ch_size!=3:
        imsize = 256
    else:
        imsize = args.imsize
    if img_transform is not None:
        image_transform = img_transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip(),
            ])
    if obj_transform is not None:
        obj_transform = obj_transform
    else:
        obj_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip()]
        )
    from lib.datasets import TextImgDataset as Dataset
    dataset = Dataset(split=split, img_transform=image_transform, obj_transform=obj_transform, args=args)
    return dataset


def prepare_datasets(args, img_transform, obj_transform):
    # train dataset
    train_dataset = prepare_dataset(args, split='train', img_transform=img_transform, obj_transform=obj_transform)
    # test dataset
    val_dataset = prepare_dataset(args, split='test', img_transform=img_transform, obj_transform=obj_transform)
    return train_dataset, val_dataset


def prepare_dataloaders(args, img_transform=None, obj_transform=None):
    batch_size = args.batch_size
    num_workers = args.num_workers
    train_dataset, valid_dataset = prepare_datasets(args, img_transform, obj_transform)
    # train dataloader
    if args.multi_gpus==True:
        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, drop_last=True,
            num_workers=num_workers, sampler=train_sampler)
    else:
        train_sampler = None
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, drop_last=True,
            num_workers=num_workers, shuffle='True')
    # valid dataloader
    if args.multi_gpus==True:
        valid_sampler = DistributedSampler(valid_dataset)
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, drop_last=True,
            num_workers=num_workers, sampler=valid_sampler)
    else:
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, drop_last=True,
            num_workers=num_workers, shuffle='True')
    return train_dataloader, valid_dataloader, \
            train_dataset, valid_dataset, train_sampler

