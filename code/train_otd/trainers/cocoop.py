import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class ImgEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.conv1 = model.conv1
        self.class_embedding = model.class_embedding
        self.positional_embedding = model.positional_embedding
        self.ln_pre = model.ln_pre
        self.transformer = model.transformer
        self.ln_post = model.ln_post
        self.proj = model.proj

    @property
    def dtype(self):
        return self.conv1.weight.dtype

    def forward(self, img: torch.Tensor):
        x = img.type(self.dtype)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        grid = x.size(-1)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        # NLD -> LND
        x = x.permute(1, 0, 2)
        # Local features
        # selected = [1,4,7,12]
        selected = [1,3,5,7,9,11]
        local_features = []
        for i in range(12):
            x = self.transformer.resblocks[i](x)
            if i in selected:
                local_features.append(
                    x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(-1, 768, grid, grid).contiguous().type(
                        img.dtype))
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return torch.stack(local_features, dim=1), x.type(img.dtype)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tkn_prompts, labels):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if labels == None:
            tokenized_prompts = tkn_prompts
        else:
            tokenized_prompts = torch.stack([tkn_prompts[l] for l in labels])
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


# class PromptLearner(nn.Module):
#     def __init__(self, cfg, classnames, clip_model):
#         super().__init__()
#         n_cls = len(classnames)
#         n_ctx = cfg.TRAINER.COCOOP.N_CTX
#         ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
#         dtype = clip_model.dtype
#         ctx_dim = clip_model.ln_final.weight.shape[0]
#         vis_dim = clip_model.visual.output_dim
#         clip_imsize = clip_model.visual.input_resolution
#         cfg_imsize = cfg.INPUT.SIZE[0]
#         assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
#
#         if ctx_init:
#             # use given words to initialize context vectors
#             ctx_init = ctx_init.replace("_", " ")
#             n_ctx = len(ctx_init.split(" "))
#             prompt = clip.tokenize(ctx_init)
#             with torch.no_grad():
#                 embedding = clip_model.token_embedding(prompt).type(dtype)
#             ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
#             prompt_prefix = ctx_init
#         else:
#             # random initialization
#             # ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
#             ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
#             nn.init.normal_(ctx_vectors, std=0.02)
#             prompt_prefix = " ".join(["X"] * n_ctx)
#
#         print(f'Initial context: "{prompt_prefix}"')
#         print(f"Number of context words (tokens): {n_ctx}")
#
#         self.ctx = nn.Parameter(ctx_vectors)
#
#         self.meta_net = nn.ModuleList([])
#         self.meta_conv = nn.ModuleList([])
#         for i in range(n_ctx):
#             self.meta_conv.append(nn.Sequential(
#                 OrderedDict([("Conv_{}".format(i), nn.Conv2d(768, 16, 1, 1, 0))])
#             ))
#             self.meta_net.append(nn.Sequential(OrderedDict([
#                 ("linear_{}_1".format(i), nn.Linear(784, 49)),
#                 ("relu_{}".format(i), nn.ReLU(inplace=True)),
#                 ("linear_{}_2".format(i), nn.Linear(49, ctx_dim))
#             ])))
#
#         if cfg.TRAINER.COCOOP.PREC == "fp16":
#             for i in range(n_ctx):
#                 self.meta_net[i].half()
#                 self.meta_conv[i].half()
#
#         classnames = [name.replace("_", " ") for name in classnames]
#         name_lens = [len(_tokenizer.encode(name)) for name in classnames]
#         prompts = [prompt_prefix + " " + name + "." for name in classnames]
#
#         tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
#         with torch.no_grad():
#             embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
#
#         # These token vectors will be saved when in save_model(),
#         # but they should be ignored in load_model() as we want to use
#         # those computed using the current class names
#         self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
#         self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
#
#         self.n_cls = n_cls
#         self.n_ctx = n_ctx
#         self.tokenized_prompts = tokenized_prompts  # torch.Tensor
#         self.name_lens = name_lens
#
#     def construct_prompts(self, ctx, prefix, suffix, label=None):
#         # dim0 is either batch_size (during training) or n_cls (during testing)
#         # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
#         # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
#         # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)
#
#         if label is not None:
#             prefix = prefix[label]
#             suffix = suffix[label]
#
#         prompts = torch.cat(
#             [
#                 prefix,  # (dim0, 1, dim)
#                 ctx,     # (dim0, n_ctx, dim)
#                 suffix,  # (dim0, *, dim)
#             ],
#             dim=1,
#         )
#
#         return prompts
#
#     def forward(self, im_features):
#         prefix = self.token_prefix
#         suffix = self.token_suffix
#         ctx = self.ctx                     # (n_cls, n_ctx, ctx_dim)
#         bias = torch.stack([self.meta_conv[i](im_features[:, i, :, :, :]).view(im_features.shape[0], -1) for i in range(self.n_ctx)], dim=1)
#         bias = torch.stack([self.meta_net[i](bias[:, i, :]) for i in range(self.n_ctx)], dim=1)   # (batch, n_ctx, ctx_dim)
#         # bias = self.meta_net(im_features)  # (batch, ctx_dim)
#         bias = bias.unsqueeze(1)           # (batch, 1, n_ctx, ctx_dim)
#         ctx = ctx.unsqueeze(0)             # (1, n_cls, n_ctx, ctx_dim)
#         ctx_shifted = ctx + bias           # (batch, n_cls, n_ctx, ctx_dim)
#
#         # Use instance-conditioned context tokens for all classes
#         prompts = []
#         for ctx_i in ctx_shifted:
#             # ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
#             pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
#             prompts.append(pts_i)
#         prompts = torch.stack(prompts)
#
#         return prompts


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COCOOP.N_CTX
        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            # ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)  # class invariant context
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx) #<--------------------------------------
            # prompt_prefix = " ".join(["X"] * 2*n_ctx) #<--------------------------------------

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}") #<--------------------------------------
        # print(f"Number of context words (tokens): {2*n_ctx}")  #<--------------------------------------

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.ModuleList([])
        self.meta_conv = nn.ModuleList([])
        for i in range(n_ctx):
            self.meta_conv.append(nn.Sequential(
                OrderedDict([("Conv_{}".format(i), nn.Conv2d(768, 16, 1, 1, 0))])  # 1*1 conv to lower the dimension
            ))
            self.meta_net.append(nn.Sequential(OrderedDict([
                ("linear_{}_1".format(i), nn.Linear(784, 49)),  # 784=7*7*16
                ("relu_{}".format(i), nn.ReLU(inplace=True)),
                ("linear_{}_2".format(i), nn.Linear(49, ctx_dim))
            ])))

        if cfg.TRAINER.COCOOP.PREC == "fp16":
            for i in range(n_ctx):
                self.meta_net[i].half()
                self.meta_conv[i].half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS #<--------------------------------------
        # self.register_buffer("token_suffix", embedding[:, 1 + 2*n_ctx :, :])  # CLS, EOS  #<--------------------------------------

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label].unsqueeze(0)
            suffix = suffix[label].unsqueeze(0)

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features, labels):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx                     # (n_cls, n_ctx, ctx_dim)
        bias = torch.stack([self.meta_conv[i](im_features[:, i, :, :, :]).view(im_features.shape[0], -1) for i in range(self.n_ctx)], dim=1)
        bias = torch.stack([self.meta_net[i](bias[:, i, :]) for i in range(self.n_ctx)], dim=1)   # (batch, n_ctx, ctx_dim)

        bias = bias.unsqueeze(1)  # (batch, 1, n_ctx, ctx_dim) --> object specific contexts
        ctx = ctx.unsqueeze(0)  # (1, n_cls, n_ctx, ctx_dim) --> class invariant contexts
        ctx_shifted = ctx + bias  # (batch, n_cls, n_ctx, ctx_dim)


        # bias = bias.unsqueeze(1).repeat(1, self.n_cls, 1, 1)  # (batch, n_cls, n_ctx, ctx_dim)  #<--------------------------------------
        # ctx = ctx.unsqueeze(0).repeat(bias.shape[0], 1, 1, 1)  # (batch, n_cls, n_ctx, ctx_dim)  #<--------------------------------------
        # ctx_shifted = torch.cat([ctx, bias], dim=2)           # (batch, n_cls, 2*n_ctx, ctx_dim)  #<--------------------------------------

        # Use instance-conditioned context tokens for all classes
        prompts = []
        if labels == None:
            for i in range(ctx_shifted.size(0)):
                # ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
                pts_i = self.construct_prompts(ctx_shifted[i], prefix, suffix)
                prompts.append(pts_i)
            prompts = torch.stack(prompts)
        else:
            for i in range(ctx_shifted.size(0)):
                label = labels[i]
                pts_i = self.construct_prompts(ctx_shifted[i][label].unsqueeze(0), prefix, suffix, label)  # (n_cls, n_tkn, ctx_dim)
                prompts.append(pts_i)
            prompts = torch.cat(prompts, dim=0)

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = ImgEncoder(clip_model.visual)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features, final_features = self.image_encoder(image.type(self.dtype))
        final_features = final_features / final_features.norm(dim=-1, keepdim=True)


        logits = []

        if self.prompt_learner.training:
            prompts = self.prompt_learner(image_features, label)
            text_features = self.text_encoder(prompts, tokenized_prompts, label)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # logits = logit_scale * final_features @ text_features.t()
            # lb = torch.arange(image_features.size(0)).to(image.device)
            # return (F.cross_entropy(logits, lb) + F.cross_entropy(logits.t(), lb))/2.0
            logits = torch.matmul(text_features, final_features.t()) * logit_scale
            text_loss = F.cross_entropy(logits, torch.arange(len(logits),device=logits.device))
            image_loss = F.cross_entropy(logits.t(), torch.arange(len(logits),device=logits.device))
            return (text_loss + image_loss) / 2.0

        else:
            prompts = self.prompt_learner(image_features, None)
            for pts_i, imf_i in zip(prompts, final_features):
                text_features = self.text_encoder(pts_i, tokenized_prompts, label)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                l_i = logit_scale * imf_i @ text_features.t()
                logits.append(l_i)
            logits = torch.stack(logits)
            return logits


@TRAINER_REGISTRY.register()
class CoCoOp(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COCOOP.PREC == "fp32" or cfg.TRAINER.COCOOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        print(f"loading model to : {self.device}")
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.COCOOP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        # impath = batch["impath"]
        # print(label)
        # print(impath)
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
