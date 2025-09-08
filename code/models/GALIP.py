####### 生成器，判别器未采用残差
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from lib.utils import dummy_context_mgr
from einops import rearrange, pack, unpack, repeat, reduce
from functools import partial
import clip


class PromptLearner(nn.Module):
    def __init__(self, n_cls, n_ctx, ctx_dim):
        super().__init__()
        dtype = torch.float16

        ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        self.meta_net = nn.ModuleList([])
        self.meta_conv = nn.ModuleList([])
        for i in range(n_ctx):
            self.meta_conv.append(nn.Sequential(
                OrderedDict([("Conv_{}".format(i), nn.Conv2d(768, 16, 1, 1, 0))])
            ))
            self.meta_net.append(nn.Sequential(OrderedDict([
                ("linear_{}_1".format(i), nn.Linear(784, 49)),
                ("relu_{}".format(i), nn.ReLU(inplace=True)),
                ("linear_{}_2".format(i), nn.Linear(49, ctx_dim))
            ])))

        self.token_prefix = nn.Parameter(torch.empty(n_cls, 1, ctx_dim, dtype=dtype))
        self.token_suffix = nn.Parameter(torch.empty(n_cls, 76 - n_ctx, ctx_dim, dtype=dtype))  # <--------------------
        # self.token_suffix = nn.Parameter(torch.empty(n_cls, 76 - 2*n_ctx, ctx_dim, dtype=dtype))

        self.n_cls = n_cls
        self.n_ctx = n_ctx

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
        ctx = self.ctx  # (n_cls, n_ctx, ctx_dim)
        bias = torch.stack(
            [self.meta_conv[i](im_features[:, i, :, :, :]).view(im_features.shape[0], -1) for i in range(self.n_ctx)],
            dim=1)
        bias = torch.stack([self.meta_net[i](bias[:, i, :]) for i in range(self.n_ctx)],
                           dim=1)  # (batch, n_ctx, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, n_ctx, ctx_dim) --> object specific contexts  # <--------------------
        ctx = ctx.unsqueeze(0)  # (1, n_cls, n_ctx, ctx_dim) --> class invariant contexts  # <--------------------
        ctx_shifted = ctx + bias # (batch, n_cls, n_ctx, ctx_dim)  # <--------------------

        # bias = bias.unsqueeze(1).repeat(1, self.n_cls, 1, 1)  # (batch, n_cls, n_ctx, ctx_dim)
        # ctx = ctx.unsqueeze(0).repeat(bias.shape[0], 1, 1, 1)  # (batch, n_cls, n_ctx, ctx_dim)
        # ctx_shifted = torch.cat([ctx, bias], dim=2)           # (batch, n_cls, 2*n_ctx, ctx_dim)

        # Use instance-conditioned context tokens for all classes
        prompts = []
        for i in range(ctx_shifted.size(0)):
            label = labels[i]
            pts_i = self.construct_prompts(ctx_shifted[i][label].unsqueeze(0), prefix, suffix,
                                           label)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.cat(prompts, dim=0)

        return prompts


class CLIP_OBJ_ENCODER(nn.Module):
    def __init__(self, CLIP):
        super(CLIP_OBJ_ENCODER, self).__init__()
        model = CLIP.visual
        # print(model)
        self.define_module(model)
        for param in self.parameters():
            param.requires_grad = False

    def define_module(self, model):
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

    def transf_to_CLIP_input(self, inputs):
        device = inputs.device
        if len(inputs.size()) != 4:
            raise ValueError('Expect the (B, C, X, Y) tensor.')
        else:
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]) \
                .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
            var = torch.tensor([0.26862954, 0.26130258, 0.27577711]) \
                .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
            inputs = F.interpolate(inputs * 0.5 + 0.5, size=(224, 224))
            inputs = ((inputs + 1) * 0.5 - mean) / var
            return inputs

    def forward(self, obj: torch.Tensor):
        x = self.transf_to_CLIP_input(obj)
        x = x.type(self.dtype)
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

        selected = [1,3,5,7,9,11]  # <<---------------------------------------------------------------------------------

        local_features = []
        for i in range(12):
            x = self.transformer.resblocks[i](x)
            if i in selected:
                local_features.append(
                    x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(-1, 768, grid, grid).contiguous().type(
                        obj.dtype))

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        # if self.proj is not None:
        #     x = x @ self.proj
        return torch.stack(local_features, dim=1), x.type(obj.dtype)


class CLIP_IMG_ENCODER(nn.Module):
    def __init__(self, CLIP):
        super(CLIP_IMG_ENCODER, self).__init__()
        model = CLIP.visual
        # print(model)
        self.define_module(model)
        for param in self.parameters():
            param.requires_grad = False

    def define_module(self, model):
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

    def transf_to_CLIP_input(self,inputs):
        device = inputs.device
        if len(inputs.size()) != 4:
            raise ValueError('Expect the (B, C, X, Y) tensor.')
        else:
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])\
                .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
            var = torch.tensor([0.26862954, 0.26130258, 0.27577711])\
                .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
            inputs = F.interpolate(inputs*0.5+0.5, size=(224, 224))
            inputs = ((inputs+1)*0.5-mean)/var
            return inputs

    def forward(self, img: torch.Tensor):
        x = self.transf_to_CLIP_input(img)
        x = x.type(self.dtype)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        grid =  x.size(-1)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        # NLD -> LND
        x = x.permute(1, 0, 2)
        # Local features
        #selected = [1,4,7,12]
        selected = [1,4,8]
        local_features = []
        for i in range(12):
            x = self.transformer.resblocks[i](x)
            if i in selected:
                local_features.append(x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(-1, 768, grid, grid).contiguous().type(img.dtype))
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return torch.stack(local_features, dim=1), x.type(img.dtype)


class CLIP_TXT_ENCODER(nn.Module):
    def __init__(self, CLIP):
        super(CLIP_TXT_ENCODER, self).__init__()
        self.define_module(CLIP)
        # print(model)
        for param in self.parameters():
            param.requires_grad = False

    def define_module(self, CLIP):
        self.transformer = CLIP.transformer
        self.vocab_size = CLIP.vocab_size
        self.token_embedding = CLIP.token_embedding
        self.positional_embedding = CLIP.positional_embedding
        self.ln_final = CLIP.ln_final
        self.text_projection = CLIP.text_projection

    @property
    def dtype(self):
        return self.transformer.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_tfoken is the highest number in each sequence)
        sent_emb = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return sent_emb, x
#
#
# class TXT_ENCODER_P(nn.Module):
#     def __init__(self, clip_model, n_ctx, cname2lab):
#         super(TXT_ENCODER_P, self).__init__()
#         self.define_module(clip_model)
#         # print(model)
#         for param in self.parameters():
#             param.requires_grad = False
#         container = set()
#         for key, values in cname2lab.items():
#             container.add((key, values))
#         mapping = {values: key for key, values in container}
#         labels = list(mapping.keys())
#         labels.sort()
#         classnames = [mapping[label] for label in labels]
#         classnames = [name.replace("_", " ") for name in classnames]
#         prompt_prefix = " ".join(["X"] * n_ctx)  # <--------------------
#         # prompt_prefix = "A photo of"  # ########################
#         # prompt_prefix = " ".join(["X"] * 2 * n_ctx)
#         prompts = [prompt_prefix + " " + name + "." for name in classnames]
#         self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
#
#     def define_module(self, CLIP):
#         self.transformer = CLIP.transformer
#         self.vocab_size = CLIP.vocab_size
#         self.token_embedding = CLIP.token_embedding
#         self.positional_embedding = CLIP.positional_embedding
#         self.ln_final = CLIP.ln_final
#         self.text_projection = CLIP.text_projection
#
#     @property
#     def dtype(self):
#         return self.transformer.resblocks[0].mlp.c_fc.weight.dtype
#
#     def forward(self, prompts, labels):
#         tokenized = torch.stack([self.tokenized_prompts[l] for l in labels])
#         # tokenized = tokenized.to(prompts.device) # ##############
#         # x = self.token_embedding(tokenized).type(self.dtype)
#         x = prompts.type(self.dtype)
#         x = x + self.positional_embedding.type(self.dtype)
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.ln_final(x).type(self.dtype)
#
#         # x.shape = [batch_size, n_ctx, transformer.width]
#         # take features from the eot embedding (eot_token is the highest number in each sequence)
#         x = x[torch.arange(x.shape[0]), tokenized.argmax(dim=-1)] @ self.text_projection
#
#         return x


class CLIP_TXT_ENCODER_WITH_OBJ(nn.Module):
    def __init__(self, CLIP, n_ctx, cname2lab):
        super(CLIP_TXT_ENCODER_WITH_OBJ, self).__init__()
        self.define_module(CLIP)
        container = set()
        for key, values in cname2lab.items():
            container.add((key, values))
        mapping = {values: key for key, values in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        classnames = [name.replace("_", " ") for name in classnames]
        prompt_prefix = " ".join(["X"] * n_ctx)
        # prompt_prefix = " ".join(["X"] * 2*n_ctx)  # <--------------------
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        # print(model)
        for param in self.parameters():
            param.requires_grad = False

    def define_module(self, CLIP):
        self.transformer = CLIP.transformer
        self.vocab_size = CLIP.vocab_size
        self.token_embedding = CLIP.token_embedding
        self.positional_embedding = CLIP.positional_embedding
        self.ln_final = CLIP.ln_final
        self.text_projection = CLIP.text_projection

    @property
    def dtype(self):
        return self.transformer.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, text, obj_txt_prompt, obj_labs):
        t = self.token_embedding(text).type(self.dtype)  # [batch_size, 77b, d_model]
        obj_txt_prompt = obj_txt_prompt.type(self.dtype)
        device = text.device
        tokenized = torch.stack([self.tokenized_prompts[l] for l in obj_labs]).to(device)
        fuse_t = []
        ori_text_len = text.argmax(dim=-1)
        for i in range(t.shape[0]):
            fuse_t.append(torch.cat((t[i, :(int(ori_text_len[i])), :], obj_txt_prompt[i, 1:78-(int(ori_text_len[i])), :]), dim=0))
        x = torch.stack(fuse_t, dim=0)
        assert x.shape[1] == 77
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_tfoken is the highest number in each sequence)
        sent_emb = x[torch.arange(x.shape[0]), text.argmax(dim=-1)+tokenized.argmax(dim=-1)-1] @ self.text_projection
        return sent_emb, x


class CLIP_Mapper(nn.Module):
    def __init__(self, CLIP):
        super(CLIP_Mapper, self).__init__()
        model = CLIP.visual
        # print(model)
        self.define_module(model)
        for param in model.parameters():
            param.requires_grad = False

    def define_module(self, model):
        self.conv1 = model.conv1
        self.class_embedding = model.class_embedding
        self.positional_embedding = model.positional_embedding
        self.ln_pre = model.ln_pre
        self.transformer = model.transformer

    @property
    def dtype(self):
        return self.conv1.weight.dtype

    def forward(self, img: torch.Tensor, prompts: torch.Tensor):
        x = img.type(self.dtype)
        prompts = prompts.type(self.dtype)
        grid = x.size(-1)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        # NLD -> LND
        x = x.permute(1, 0, 2)
        # Local features
        selected = [1,2,3,4,5,6,7,8]
        begin, end = 0, 12
        prompt_idx = 0
        for i in range(begin, end):
            if i in selected:
                prompt = prompts[:,prompt_idx,:].unsqueeze(0)
                prompt_idx = prompt_idx+1
                x = torch.cat((x,prompt), dim=0)
                x = self.transformer.resblocks[i](x)
                x = x[:-1,:,:]
            else:
                x = self.transformer.resblocks[i](x)
        return x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(-1, 768, grid, grid).contiguous().type(img.dtype)


class CLIP_Adapter(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, G_ch, CLIP_ch, cond_dim, k, s, p, map_num, CLIP):
        super(CLIP_Adapter, self).__init__()
        self.CLIP_ch = CLIP_ch
        self.FBlocks = nn.ModuleList([])
        self.FBlocks.append(M_Block(in_ch, mid_ch, out_ch, cond_dim, k, s, p))
        for i in range(map_num-1):
            self.FBlocks.append(M_Block(out_ch, mid_ch, out_ch, cond_dim, k, s, p))
        self.conv_fuse = nn.Conv2d(out_ch, CLIP_ch, 5, 1, 2)
        self.CLIP_ViT = CLIP_Mapper(CLIP)
        self.conv = nn.Conv2d(768, G_ch, 5, 1, 2)
        #
        self.fc_prompt = nn.Linear(cond_dim, CLIP_ch*8)

    def forward(self,out,c):
        prompts = self.fc_prompt(c).view(c.size(0),-1,self.CLIP_ch)
        for FBlock in self.FBlocks:
            out = FBlock(out,c)
        fuse_feat = self.conv_fuse(out)
        map_feat = self.CLIP_ViT(fuse_feat,prompts)
        return self.conv(fuse_feat+0.1*map_feat)


class NetG(nn.Module):
    def __init__(self, ngf, nz, cond_dim, imsize, ch_size, mixed_precision, CLIP):
        super(NetG, self).__init__()
        self.ngf = ngf
        self.mixed_precision = mixed_precision
        # build CLIP Mapper
        self.code_sz, self.code_ch, self.mid_ch = 7, 64, 32
        self.CLIP_ch = 768
        self.aug_2 = 0.75
        self.fc_code = nn.Linear(nz, self.code_sz*self.code_sz*self.code_ch)
        self.mapping = CLIP_Adapter(self.code_ch, self.mid_ch, self.code_ch, ngf*8, self.CLIP_ch, cond_dim+nz, 3, 1, 1, 4, CLIP)  # <----------------------
        # build GBlocks
        self.GBlocks = nn.ModuleList([])
        in_out_pairs = list(get_G_in_out_chs(ngf, imsize))
        imsize = 4
        for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
            if idx<(len(in_out_pairs)-1):
                imsize = imsize*2
            else:
                imsize = 224
            self.GBlocks.append(G_Block(cond_dim+nz, in_ch, out_ch, imsize))  # <------------------------------------
        # to RGB image
        self.to_rgb = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(out_ch, ch_size, 3, 1, 1),
            #nn.Tanh(),
            )
        # self.proj = nn.Linear(768, 512)  # <<<<<<<

    def forward(self, noise, c, eval=False): # x=noise, c=ent_emb
        with torch.cuda.amp.autocast() if self.mixed_precision and not eval else dummy_context_mgr() as mp:
            # o = self.proj(o)  # <<<<<<<<<
            cond = torch.cat((noise, c), dim=1)  # <--------------------------------------------------------
            out = self.mapping(self.fc_code(noise).view(noise.size(0), self.code_ch, self.code_sz, self.code_sz), cond)
            # fuse text and visual features
            for GBlock in self.GBlocks:
                out = GBlock(out, cond)
            # convert to RGB image
            out = self.to_rgb(out)
        return out


# 定义鉴别器网络D
class NetD(nn.Module):
    def __init__(self, ndf, imsize, ch_size, mixed_precision):
        super(NetD, self).__init__()
        self.mixed_precision = mixed_precision
        self.DBlocks = nn.ModuleList([
            D_Block(768, 768, 3, 1, 1, res=True, CLIP_feat=True),
            D_Block(768, 768, 3, 1, 1, res=True, CLIP_feat=True),
        ])
        self.main = D_Block(768, 512, 3, 1, 1, res=True, CLIP_feat=False)

    def forward(self, h):
        with torch.cuda.amp.autocast() if self.mixed_precision else dummy_context_mgr() as mpc:
            out = h[:,0]
            for idx in range(len(self.DBlocks)):
                out = self.DBlocks[idx](out, h[:,idx+1])
            out = self.main(out)
        return out


class NetC(nn.Module):
    def __init__(self, ndf, cond_dim, mixed_precision):
        super(NetC, self).__init__()
        self.cond_dim = cond_dim
        self.mixed_precision = mixed_precision
        self.joint_conv = nn.Sequential(
            nn.Conv2d(512+512, 128, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 4, 1, 0, bias=False),
            )

    def forward(self, out, cond):
        with torch.cuda.amp.autocast() if self.mixed_precision else dummy_context_mgr() as mpc:
            cond = cond.view(-1, self.cond_dim, 1, 1)
            cond = cond.repeat(1, 1, 7, 7)
            h_c_code = torch.cat((out, cond), 1)
            out = self.joint_conv(h_c_code)
        return out


class M_Block(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, cond_dim, k, s, p):
        super(M_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, k, s, p)
        self.fuse1 = DFBLK(cond_dim, mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, k, s, p)
        self.fuse2 = DFBLK(cond_dim, out_ch)
        self.learnable_sc = in_ch != out_ch
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, h, text):
        h = self.conv1(h)
        h = self.fuse1(h, text)
        h = self.conv2(h)
        h = self.fuse2(h, text)
        return h

    def forward(self, h, c):
        return self.shortcut(h) + self.residual(h, c)


# class M_Block_m(nn.Module):
#     def __init__(self, in_ch, mid_ch, out_ch, cond_dim, k, s, p):
#         super(M_Block_m, self).__init__()
#         self.conv1 = nn.Conv2d(in_ch, mid_ch, k, s, p)
#         self.fuse1 = DFBLK(cond_dim, mid_ch)
#         self.fuse1_space = DFBLK_m(512, 7)
#         self.conv2 = nn.Conv2d(mid_ch, out_ch, k, s, p)
#         self.fuse2 = DFBLK(cond_dim, out_ch)
#         self.fuse2_space = DFBLK_m(512, 7)
#         self.learnable_sc = in_ch != out_ch
#         if self.learnable_sc:
#             self.c_sc = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
#
#     def shortcut(self, x):
#         if self.learnable_sc:
#             x = self.c_sc(x)
#         return x
#
#     def residual(self, h, text, obj):
#         h = self.conv1(h)
#         h = self.fuse1(h, text)
#         h = self.fuse1_space(h, obj)
#         h = self.conv2(h)
#         h = self.fuse2(h, text)
#         h = self.fuse2_space(h, obj)
#         return h
#
#     def forward(self, h, c, o):
#         return self.shortcut(h) + self.residual(h, c, o)


class G_Block(nn.Module):
    def __init__(self, cond_dim, in_ch, out_ch, imsize):
        super(G_Block, self).__init__()
        self.imsize = imsize
        self.learnable_sc = in_ch != out_ch 
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.fuse1 = DFBLK(cond_dim, in_ch)
        self.fuse2 = DFBLK(cond_dim, out_ch)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch,out_ch, 1, stride=1, padding=0)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, h, y):
        h = self.fuse1(h, y)
        h = self.c1(h)
        h = self.fuse2(h, y)
        h = self.c2(h)
        return h

    def forward(self, h, y):
        h = F.interpolate(h, size=(self.imsize, self.imsize))
        return self.shortcut(h) + self.residual(h, y)



# class G_Block_m(nn.Module):
#     def __init__(self, cond_dim, in_ch, out_ch, imsize):
#         super(G_Block_m, self).__init__()
#         self.imsize = imsize
#         self.learnable_sc = in_ch != out_ch
#         self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
#         self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
#         self.fuse1 = DFBLK(cond_dim, in_ch)
#         self.fuse2 = DFBLK(cond_dim, out_ch)
#         self.xattn = CrossAttentionBlock(dim=out_ch, ff_mult=2)
#         self.o_conv = nn.Conv2d(768, 32, 1, 1, 0)
#         self.o_fc = nn.Linear(7*7*32, 512)
#         if self.learnable_sc:
#             self.c_sc = nn.Conv2d(in_ch,out_ch, 1, stride=1, padding=0)
#
#     def shortcut(self, x):
#         if self.learnable_sc:
#             x = self.c_sc(x)
#         return x
#
#     def residual(self, h, y, o_t, o_v):
#         h = self.fuse1(h, y)
#         h = self.c1(h)
#         h = self.fuse2(h, y)
#         h = self.c2(h)
#         # h = self.sattn(h)
#         o_v = self.o_conv(o_v)
#         o_v = self.o_fc(o_v.view(h.shape[0], -1))
#         h = self.xattn(h, o_t, o_v)
#         return h
#
#     def forward(self, h, y, o_t, o_v):
#         h = F.interpolate(h, size=(self.imsize, self.imsize))
#         return self.shortcut(h) + self.residual(h, y, o_t, o_v)




class D_Block(nn.Module):
    def __init__(self, fin, fout, k, s, p, res, CLIP_feat):
        super(D_Block, self).__init__()
        self.res, self.CLIP_feat = res, CLIP_feat
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, k, s, p, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fout, fout, k, s, p, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            )
        self.conv_s = nn.Conv2d(fin, fout, 1, stride=1, padding=0)
        if self.res==True:
            self.gamma = nn.Parameter(torch.zeros(1))
        if self.CLIP_feat==True:
            self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x, CLIP_feat=None):
        res = self.conv_r(x)
        if self.learned_shortcut:
            x = self.conv_s(x)
        if (self.res==True)and(self.CLIP_feat==True):
            return x + self.gamma*res + self.beta*CLIP_feat
        elif (self.res==True)and(self.CLIP_feat!=True):
            return x + self.gamma*res
        elif (self.res!=True)and(self.CLIP_feat==True):
            return x + self.beta*CLIP_feat
        else:
            return x


class DFBLK(nn.Module):
    def __init__(self, cond_dim, in_ch):
        super(DFBLK, self).__init__()
        self.affine0 = Affine(cond_dim, in_ch)
        self.affine1 = Affine(cond_dim, in_ch)

    def forward(self, x, y=None):
        h = self.affine0(x, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        h = self.affine1(h, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        return h


# class DFBLK_m(nn.Module):
#     def __init__(self, cond_dim, imsize):
#         super(DFBLK_m, self).__init__()
#         self.affine0 = Affine_space(cond_dim, imsize)
#         self.affine1 = Affine_space(cond_dim, imsize)
#
#     def forward(self, x, y=None):
#         h = self.affine0(x, y)
#         h = nn.LeakyReLU(0.2, inplace=True)(h)
#         h = self.affine1(h, y)
#         h = nn.LeakyReLU(0.2, inplace=True)(h)
#         return h


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Affine(nn.Module):
    def __init__(self, cond_dim, num_features):
        super(Affine, self).__init__()

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, num_features)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(num_features, num_features)),
            ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, num_features)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(num_features, num_features)),
            ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None):
        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)        

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias


# class Affine_space(nn.Module):
#     def __init__(self, cond_dim, grid):
#         super(Affine_space, self).__init__()
#
#         self.fc_gamma = nn.Sequential(OrderedDict([
#             ('linear1', nn.Linear(cond_dim, grid*grid)),
#             ('relu1', nn.ReLU(inplace=True)),
#             ('linear2', nn.Linear(grid*grid, grid*grid)),
#         ]))
#         self.fc_beta = nn.Sequential(OrderedDict([
#             ('linear1', nn.Linear(cond_dim, grid*grid)),
#             ('relu1', nn.ReLU(inplace=True)),
#             ('linear2', nn.Linear(grid*grid, grid*grid)),
#         ]))
#         self._initialize()
#
#     def _initialize(self):
#         nn.init.zeros_(self.fc_gamma.linear2.weight.data)
#         nn.init.ones_(self.fc_gamma.linear2.bias.data)
#         nn.init.zeros_(self.fc_beta.linear2.weight.data)
#         nn.init.zeros_(self.fc_beta.linear2.bias.data)
#
#     def forward(self, x, y=None):
#         weight = self.fc_gamma(y).view(y.shape[0], x.shape[-1], x.shape[-1])
#         bias = self.fc_beta(y).view(y.shape[0], x.shape[-1], x.shape[-1])
#
#         # if weight.dim() == 1:
#         #     weight = weight.unsqueeze(0)
#         # if bias.dim() == 1:
#         #     bias = bias.unsqueeze(0)
#
#         size = x.size()
#         weight = weight.unsqueeze(1).expand(size)
#         bias = bias.unsqueeze(1).expand(size)
#         return weight * x + bias


def get_G_in_out_chs(nf, imsize):
    layer_num = int(np.log2(imsize))-1
    channel_nums = [nf*min(2**idx, 8) for idx in range(layer_num)]
    channel_nums = channel_nums[::-1]
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
    return in_out_pairs


def get_D_in_out_chs(nf, imsize):
    layer_num = int(np.log2(imsize))-1
    channel_nums = [nf*min(2**idx, 8) for idx in range(layer_num)]
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
    return in_out_pairs


# class ChannelRMSNorm(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.scale = dim ** 0.5
#         self.gamma = nn.Parameter(torch.ones(dim, 1, 1))
#
#     def forward(self, x):
#         normed = F.normalize(x, dim = 1)
#         return normed * self.scale * self.gamma
#
#
# class RMSNorm(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.scale = dim ** 0.5
#         self.gamma = nn.Parameter(torch.ones(dim))
#
#     def forward(self, x):
#         normed = F.normalize(x, dim = -1)
#         return normed * self.scale * self.gamma
#
#
# def exists(val):
#     return val is not None
#
#
# def default(*vals):
#     for val in vals:
#         if exists(val):
#             return val
#     return None
#
#
# def FeedForward(
#     dim,
#     mult = 4,
#     channel_first = False
# ):
#     dim_hidden = int(dim * mult)
#     norm_klass = ChannelRMSNorm if channel_first else RMSNorm
#     proj = partial(nn.Conv2d, kernel_size = 1) if channel_first else nn.Linear
#
#     return nn.Sequential(
#         norm_klass(dim),
#         proj(dim, dim_hidden),
#         nn.GELU(),
#         proj(dim_hidden, dim)
#     )
#
#
# class CrossAttention(nn.Module):
#     def __init__(
#         self,
#         dim,
#         dim_head = 64,
#         heads = 8
#     ):
#         super().__init__()
#         self.heads = heads
#         self.scale = dim_head ** -0.5
#         dim_inner = dim_head * heads
#         k_input_dim = 512
#         v_input_dim = 512
#
#         self.norm = ChannelRMSNorm(dim)
#         self.norm_context = RMSNorm(k_input_dim)
#         self.norm_visual = RMSNorm(v_input_dim)
#
#         self.to_q = nn.Conv2d(dim, dim_inner, 1, bias = False)
#         self.to_k = nn.Linear(k_input_dim, dim_inner, bias = False)
#         self.to_v = nn.Linear(v_input_dim, dim_inner, bias = False)
#         self.to_out = nn.Conv2d(dim_inner, dim, 1, bias = False)
#
#     def forward(self, fmap, context, visual):
#         """
#         einstein notation
#
#         b - batch
#         h - heads
#         x - height
#         y - width
#         d - dimension
#         i - source seq (attend from)
#         j - target seq (attend to)
#         """
#
#         if len(context.size()) < 3:
#             context = context.unsqueeze(1)
#         if len(visual.size()) < 3:
#             visual = visual.unsqueeze(1)
#
#         fmap = self.norm(fmap)
#         context = self.norm_context(context)
#         visual = self.norm_visual(visual)
#
#         x, y = fmap.shape[-2:]
#
#         h = self.heads
#
#         q, k, v = (self.to_q(fmap), self.to_k(context), self.to_v(visual))
#
#         k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (k, v))
#
#         q = rearrange(q, 'b (h d) x y -> (b h) (x y) d', h = self.heads)
#
#         sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
#
#         attn = sim.softmax(dim = -1)
#
#         out = torch.einsum('b i j, b j d -> b i d', attn, v)
#
#         out = rearrange(out, '(b h) (x y) d -> b (h d) x y', x = x, y = y, h = h)
#
#         return self.to_out(out)
#
#
# class CrossAttentionBlock(nn.Module):
#     def __init__(
#         self,
#         dim,
#         dim_head = 64,
#         heads = 8,
#         ff_mult = 4
#     ):
#         super().__init__()
#         self.attn = CrossAttention(dim = dim, dim_head = dim_head, heads = heads)
#         self.ff = FeedForward(dim = dim, mult = ff_mult, channel_first = True)
#         self.apply(self.init_)
#
#     def init_(self, m):
#             if type(m) in {nn.Conv2d, nn.Linear}:
#                 nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
#
#     def forward(self, x, context, visual):
#         x = self.attn(x, context = context, visu