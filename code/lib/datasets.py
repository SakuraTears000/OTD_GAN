import os
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
import clip as clip
# import open_clip


def get_fix_data(train_dl, test_dl, text_encoder, text_encoder_obj, obj_encoder, PL, args):
    fixed_image_train, fixed_obj_f_train, fixed_obj_tk_train, _, _, fixed_sent_train, fixed_word_train, fixed_sent_with_obj_train, fixed_key_train = get_one_batch_data(train_dl, text_encoder, text_encoder_obj, obj_encoder, PL, args)
    fixed_image_test, fixed_obj_f_test, fixed_obj_tk_test, _, _, fixed_sent_test, fixed_word_test, fixed_sent_with_obj_test, fixed_key_test = get_one_batch_data(test_dl, text_encoder, text_encoder_obj, obj_encoder, PL, args)
    fixed_image = torch.cat((fixed_image_train, fixed_image_test), dim=0)
    fixed_obj_f = torch.cat((fixed_obj_f_train, fixed_obj_f_test), dim=0)
    fixed_noise = torch.randn(fixed_image.size(0), args.z_dim).to(args.device)
    fixed_noise_o = torch.randn(fixed_image.size(0), args.cond_dim).to(args.device)
    fixed_obj_tk = torch.cat((fixed_obj_tk_train, fixed_obj_tk_test), dim=0)
    fixed_sent = torch.cat((fixed_sent_train, fixed_sent_test), dim=0)
    fixed_sent_with_obj = torch.cat((fixed_sent_with_obj_train, fixed_sent_with_obj_test), dim=0)
    return fixed_image, fixed_obj_f, fixed_obj_tk, fixed_sent, fixed_sent_with_obj, fixed_noise, fixed_noise_o


def get_one_batch_data(dataloader, text_encoder, text_encoder_obj, obj_encoder, PL, args):
    data = next(iter(dataloader))
    imgs, obj_f, obj_tk, captions, CLIP_tokens, sent_emb, words_embs, sent_emb_obj, _, keys = prepare_data(data, text_encoder, text_encoder_obj, obj_encoder, PL,args.device)
    return imgs, obj_f, obj_tk, captions, CLIP_tokens, sent_emb, words_embs, sent_emb_obj, keys


def prepare_data(data, text_encoder, text_encoder_obj, obj_encoder, PL, device):
    imgs, objs, obj_labs, captions, CLIP_tokens, keys= data
    imgs, objs, obj_labs, CLIP_tokens= imgs.to(device), objs.to(device), obj_labs.to(device), CLIP_tokens.to(device)
    obj_f, obj_tk = encode_obj(obj_encoder, objs)
    obj_txt_prompts = PL(obj_f, obj_labs).detach().to(device)
    sent_emb, words_embs = encode_tokens_ori(text_encoder, CLIP_tokens)
    sent_emb_with_obj, words_embs_with_obj = encode_tokens_obj(text_encoder_obj, CLIP_tokens, obj_txt_prompts, obj_labs)
    return imgs, obj_f, obj_tk, captions, CLIP_tokens, sent_emb, words_embs, sent_emb_with_obj, words_embs_with_obj, keys


def encode_obj(obj_encoder, obj):
    # encode text
    with torch.no_grad():
        obj_f, obj_tk = obj_encoder(obj)
        obj_f, obj_tk = obj_f.detach(), obj_tk.detach()
    return obj_f, obj_tk

def encode_tokens_ori(text_encoder, caption):
    with torch.no_grad():
        sent_emb, words_embs = text_encoder(caption)
        sent_emb, words_embs = sent_emb.detach(), words_embs.detach()
    return sent_emb, words_embs

def encode_tokens_obj(text_encoder, caption, obj_txt_prompts, obj_labs):
    # encode text
    with torch.no_grad():
        sent_emb,words_embs = text_encoder(caption, obj_txt_prompts, obj_labs)
        sent_emb,words_embs = sent_emb.detach(), words_embs.detach()
    return sent_emb, words_embs


def get_imgs(img_path, bbox=None, img_transform=None, obj_transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    if bbox is not None:
        x1, y1, x2, y2 = eval(bbox[0]), eval(bbox[1]), eval(bbox[2]), eval(bbox[3]),
        obj = img.crop([x1, y1, x2, y2])
    else:
        raise AssertionError
    if img_transform is not None:
        img = img_transform(img)
    if obj_transform is not None:
        obj = obj_transform(obj)
    if normalize is not None:
        img = normalize(img)
        obj = normalize(obj)
    return img, obj


def get_caption(cap_path,clip_info):
    eff_captions = []
    with open(cap_path, "r") as f:
        captions = f.read().encode('utf-8').decode('utf8').split('\n')
    for cap in captions:
        if len(cap) != 0:
            eff_captions.append(cap)
    sent_ix = random.randint(0, len(eff_captions))
    caption = eff_captions[sent_ix]
    # tokenizer = open_clip.get_tokenizer('ViT-B-32')
    # tokens = tokenizer(caption)
    tokens = clip.tokenize(caption, truncate=True)
    return caption, tokens[0]


################################################################
#                    Dataset
################################################################
class TextImgDataset(data.Dataset):
    def __init__(self, split, img_transform=None, obj_transform=None, args=None):
        self.img_transform = img_transform
        self.obj_transform = obj_transform
        self.cname2lab = args.cname2lab
        self.clip4text = args.clip4text
        self.data_dir = args.data_dir
        self.dataset_name = args.dataset_name
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        self.split = split
        self.split_dir = os.path.join(self.data_dir, split)
        self.filenames = self.load_filenames(self.data_dir, split)
        self.number_example = len(self.filenames)

    def load_obj_bbox(self, key):
        data_dir = self.data_dir
        if self.dataset_name.lower().find('nwpu') != -1:
            bbox_path = '%s/img_for_gen/det_result/%s/%s/objects/%s.txt' % (data_dir, self.split, key.split('/')[0], key.split('/')[-1])  # for nwpu dataset
        elif self.dataset_name.lower().find('rsicd') != -1:
            bbox_path = '%s/objs/%s.txt' % (data_dir, key.split('/')[-1])
        elif self.dataset_name.find('special') != -1:
            bbox_path = '%s/bbox/%s.txt' % (data_dir, key)
        obj_count = 0
        with open(bbox_path, "r") as f:
            objs = f.read().split('\n')
        for obj in objs:
            if len(obj) != 0:
                obj_count += 1
        assert obj_count > 0
        if self.dataset_name.find('special') != -1:
            obj_ix = 0
        else:
            obj_ix = random.randint(0, obj_count)
        obj_cls = objs[obj_ix].split(' ')[0]

        # Load obj_bbox
        x1 = objs[obj_ix].split(' ')[1]
        y1 = objs[obj_ix].split(' ')[2]
        x2 = objs[obj_ix].split(' ')[3]
        y2 = objs[obj_ix].split(' ')[4]
        obj_bbox = [x1, y1, x2, y2]

        return obj_cls, obj_bbox

    def load_filenames(self, data_dir, split):
        filenames=[]
        if self.dataset_name.find('special') == -1:
            folder_path = '%s/img_for_gen/%s' % (data_dir, split)
            if os.path.isdir(folder_path):
                for c in os.listdir(folder_path):
                    for f in os.listdir(os.path.join(folder_path, c)):
                        filename = c + '/' + f.split('.')[0]
                        filenames.append(filename)
        else:
            folder_path = '%s/test_image' % data_dir
            if os.path.isdir(folder_path):
                for f in os.listdir(folder_path):
                    filename = f.split('.')[0]
                    filenames.append(filename)
        print('Load filenames from: %s (%d)' % (folder_path, len(filenames)))
        return filenames

    def __getitem__(self, index):
        #
        key = self.filenames[index]
        data_dir = self.data_dir
        #
        if self.dataset_name.lower().find('nwpu') != -1:
            if self.split=='train':
                img_name = '%s/img_for_gen/train/%s.jpg' % (data_dir, key)
                text_name = '%s/caption/%s.txt' % (data_dir, key)
            else:
                img_name = '%s/img_for_gen/test/%s.jpg' % (data_dir, key)
                text_name = '%s/caption/%s.txt' % (data_dir, key)
        elif self.dataset_name.lower().find('rsicd') != -1:
            if self.split=='train':
                img_name = '%s/img_for_gen/train/%s.jpg' % (data_dir, key)
                text_name = '%s/captions/%s.txt' % (data_dir, key.split('/')[-1])
            else:
                img_name = '%s/img_for_gen/test/%s.jpg' % (data_dir, key)
                text_name = '%s/captions/%s.txt' % (data_dir, key.split('/')[-1])
        elif self.dataset_name.find('special') != -1:
            img_name = '%s/test_image/%s.jpg' % (data_dir, key)
            text_name = '%s/test_caption/sample.txt' % data_dir
        obj_cls, obj_bbox = self.load_obj_bbox(key)
        #
        imgs, objs = get_imgs(img_name, obj_bbox, self.img_transform, self.obj_transform, normalize=self.norm)
        caps, tokens = get_caption(text_name, self.clip4text)
        obj_labs = self.cname2lab[obj_cls]
        return imgs, objs, obj_labs, caps, tokens, key

    def __len__(self):
        return len(self.filenames)

