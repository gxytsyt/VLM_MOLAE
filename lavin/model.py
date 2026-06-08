# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import sys
sys.path.append('/data/yiping/LaVIN-video-all/LaVIN-video-1expert-1-rank16-nobias-group1-video-qformer')
from video_llama.models.blip2 import Blip2Base
# from transformers import BertConfig
from video_llama.models.Qformer import BertConfig, BertLMHeadModel
import einops

from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)

from torch.nn import Embedding, Linear
import torch
import pdb
from timm.models.layers import  DropPath
import clip
from  torch.cuda.amp import autocast
@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    hidden_proj: int=128

    max_batch_size: int = 32
    max_seq_len: int = 2048
    drop_path: float=0.


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        #modified bias for reparameterizing
        self.wq = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wk = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wv = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wo = Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False
        )


    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):

        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = xk
        values = xv


        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Use PyTorch 2.x scaled_dot_product_attention (Flash / Memory Efficient backend)
        output = F.scaled_dot_product_attention(
            xq, keys, values,
            attn_mask=mask,
            is_causal=False,
            scale=1.0 / math.sqrt(self.head_dim),
        )  # (B, H, S, D)

        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = Linear(
            dim, hidden_dim, bias=False
        )
        self.w2 = Linear(
            hidden_dim, dim, bias=False
        )
        self.w3 = Linear(
            dim, hidden_dim, bias=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x),inplace=False) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.drop_path = DropPath(args.drop_path) if args.drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):

        h = x + self.drop_path(self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, adapter))
        out = h + self.drop_path(self.feed_forward.forward(self.ffn_norm(h)))
        return out



class AdapterMLP(nn.Module):
    """ Pytorch Implemention of RepAdapter for 1d tensor"""

    def __init__(
            self,
            in_features=768,
            hidden_dim=128,
            out_features=4096
    ):
        super().__init__()
        self.conv_A = nn.Linear(in_features, hidden_dim)
        self.conv_B = nn.Linear(hidden_dim, out_features)


        nn.init.xavier_uniform_( self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.xavier_uniform_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)

    def forward(self, x):
        with autocast():
            x = self.conv_B(F.silu(self.conv_A(x)))
        return x


class Transformer(nn.Module):

    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width, num_hidden_layers=2):

        encoder_config = BertConfig.from_pretrained("/data/yiping/LaVIN-video-all/bert_base_uncased")

        encoder_config.hidden_size = vision_width
        encoder_config.num_attention_heads = 16

        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = Embedding(
            params.vocab_size, params.dim
        )

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        # with init_empty_weights():
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = Linear(
            params.dim, params.vocab_size, bias=False
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

        # self.backbone = clip.load('ViT-L/14')[0]

        self.video_q_former_adapter, self.video_query_tokens_adapter = self.init_video_Qformer(num_query_token=40,
                                                                                               # vision_width=768,
                                                                                               vision_width=1024,
                                                                                               num_hidden_layers=2)
        # self.video_frame_position_embedding_adapter = nn.Embedding(40, 768)
        self.video_frame_position_embedding_adapter = nn.Embedding(40, 1024)

        print('define AdapterMLP')

        self.vid768_1024_adapter_proj = nn.Linear(768, 1024).float()

        self.mm_adapter_proj = nn.Linear(1024, params.dim).float()
        self.adapter_modality_embedding = nn.Embedding(2, params.dim).float()


    def insert_image_embeds(self,examples, labels, vid_embeds, prefix_vid, img_indicators):
        _bsz, seqlen,_ = examples.shape

        prefix_img = prefix_vid
        image_embeds = vid_embeds

        new_examples = []
        new_labels = []
        for i, (example, label) in enumerate(zip(examples, labels)):
            if img_indicators[i] > 0.:
                new_example = torch.cat([example[:1], prefix_img, image_embeds[i], example[1:]], 0)
                new_label = torch.cat([label[:1],
                                     torch.zeros(prefix_img.shape[0]+image_embeds.shape[1]).to(examples.device).type_as(labels),
                                     label[1:]])
                new_example = new_example[:seqlen]
                new_label = new_label[:seqlen]
            else:
                raise

            new_examples.append(new_example.unsqueeze(0))
            new_labels.append(new_label.unsqueeze(0))
        new_examples = torch.cat(new_examples, 0)
        new_labels = torch.cat(new_labels, 0)


        return new_examples, new_labels

    def forward(self, examples, labels, video_fea=None, prefix_vid=None, images=None, prefix_img=None, prefix_nonimg=None, img_indicators=None):
        if isinstance(img_indicators,list):
            img_indicators = torch.Tensor(img_indicators).to(image_embeds.device).long()
        modality_embed = self.adapter_modality_embedding(img_indicators.unsqueeze(1))


        with autocast():
            video_fea = self.vid768_1024_adapter_proj(video_fea)
        batch_size, time_length, frame_qfm, vis_hid = video_fea.size()

        position_ids = torch.arange(time_length).to(labels.device).type_as(labels)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        frame_position_embeddings = self.video_frame_position_embedding_adapter(position_ids)
        frame_hidden_state = video_fea
        frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
        # print(frame_position_embeddings.shape)  # torch.Size([16, 40, 1, 768])
        frame_hidden_state = frame_position_embeddings + frame_hidden_state

        frame_hidden_state = einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h', b=batch_size, t=time_length)
        frame_atts = torch.ones(frame_hidden_state.size()[:-1]).to(labels.device).type_as(labels)
        video_query_tokens = self.video_query_tokens_adapter.expand(frame_hidden_state.shape[0], -1, -1)
        video_query_output = self.video_q_former_adapter.bert(
            query_embeds=video_query_tokens,
            encoder_hidden_states=frame_hidden_state,
            encoder_attention_mask=frame_atts,
            return_dict=True,
        )
        video_hidden = video_query_output.last_hidden_state

        with autocast():
            video_embeds = self.mm_adapter_proj(video_hidden)


        _bsz, seqlen = examples.shape

        examples = self.tok_embeddings(examples)

        prefix_vid = self.tok_embeddings(prefix_vid.unsqueeze(0)).squeeze(0)

        h, labels = self.insert_image_embeds(examples, labels, video_embeds, prefix_vid, img_indicators)

        h = torch.cat([modality_embed.half(), h], 1)[:, :seqlen]
        modality_labels = torch.zeros(_bsz, 1).to(labels.device).type_as(labels)
        labels = torch.cat([modality_labels, labels], 1)[:, :seqlen]


        freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[:seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(h)  # torch.Size([1, 1, 512, 512])

        # mask decision token
        mask[:, :, 1:, 0] = float("-inf")

        start_pos = 0
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        h = self.norm(h)
        output = self.output(h)
        output = output[:, :-1, :].reshape(-1, self.vocab_size)
        labels = labels[:, 1:].flatten()


        c_loss = self.criterion(output, labels)
        return c_loss
