"""
Taken from https://github.com/lucidrains/g-mlp-pytorch

@misc{liu2021pay,
    title   = {Pay Attention to MLPs}, 
    author  = {Hanxiao Liu and Zihang Dai and David R. So and Quoc V. Le},
    year    = {2021},
    eprint  = {2105.08050},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}

@software{peng_bo_2021_5196578,
    author       = {PENG Bo},
    title        = {BlinkDL/RWKV-LM: 0.01},
    month        = aug,
    year         = 2021,
    publisher    = {Zenodo},
    version      = {0.01},
    doi          = {10.5281/zenodo.5196578},
    url          = {https://doi.org/10.5281/zenodo.5196578%7D
}
"""


from random import randrange
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

# functions


def exists(val):
    return val is not None


def pair(val):
    return (val, val) if not isinstance(val, tuple) else val


def dropout_layers(layers, prob_survival):
    if prob_survival == 1:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0.0, 1.0) > prob_survival

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers


def shift(t, amount, mask=None):
    if amount == 0:
        return t
    return F.pad(t, (0, 0, amount, -amount), value=0.0)


# helper classes


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreShiftTokens(nn.Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        if self.shifts == (0,):
            return self.fn(x, **kwargs)

        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim=-1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(map(lambda args: shift(*args), zip(segments_to_shift, shifts)))
        x = torch.cat((*segments_to_shift, *rest), dim=-1)
        return self.fn(x, **kwargs)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class Attention(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner, causal=False):
        super().__init__()
        self.scale = dim_inner ** -0.5
        self.causal = causal

        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias=False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        device = x.device
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if self.causal:
            mask = torch.ones(sim.shape[-2:], device=device).triu(1).bool()
            sim.masked_fill_(mask[None, ...], -torch.finfo(q.dtype).max)

        attn = sim.softmax(dim=-1)
        out = einsum("b i j, b j d -> b i d", attn, v)
        return self.to_out(out)


class SpatialGatingUnit(nn.Module):
    def __init__(self, dim, dim_seq, causal=False, act=nn.Identity(), heads=1, init_eps=1e-3, circulant_matrix=False):
        super().__init__()
        dim_out = dim // 2
        self.heads = heads
        self.causal = causal
        self.norm = nn.LayerNorm(dim_out)

        self.act = act

        # parameters

        if circulant_matrix:
            self.circulant_pos_x = nn.Parameter(torch.ones(heads, dim_seq))
            self.circulant_pos_y = nn.Parameter(torch.ones(heads, dim_seq))

        self.circulant_matrix = circulant_matrix
        shape = (
            (
                heads,
                dim_seq,
            )
            if circulant_matrix
            else (heads, dim_seq, dim_seq)
        )
        weight = torch.zeros(shape)

        self.weight = nn.Parameter(weight)
        init_eps /= dim_seq
        nn.init.uniform_(self.weight, -init_eps, init_eps)

        self.bias = nn.Parameter(torch.ones(heads, dim_seq))

    def forward(self, x, gate_res=None):
        device, n, h = x.device, x.shape[1], self.heads

        res, gate = x.chunk(2, dim=-1)
        gate = self.norm(gate)

        weight, bias = self.weight, self.bias

        if self.circulant_matrix:
            # build the circulant matrix

            dim_seq = weight.shape[-1]
            weight = F.pad(weight, (0, dim_seq), value=0)
            weight = repeat(weight, "... n -> ... (r n)", r=dim_seq)
            weight = weight[:, :-dim_seq].reshape(h, dim_seq, 2 * dim_seq - 1)
            weight = weight[:, :, (dim_seq - 1) :]

            # give circulant matrix absolute position awareness

            pos_x, pos_y = self.circulant_pos_x, self.circulant_pos_y
            weight = weight * rearrange(pos_x, "h i -> h i ()") * rearrange(pos_y, "h j -> h () j")

        if self.causal:
            weight, bias = weight[:, :n, :n], bias[:, :n]
            mask = torch.ones(weight.shape[-2:], device=device).triu_(1).bool()
            mask = rearrange(mask, "i j -> () i j")
            weight = weight.masked_fill(mask, 0.0)

        gate = rearrange(gate, "b n (h d) -> b h n d", h=h)

        gate = einsum("b h n d, h m n -> b h m d", gate, weight)
        gate = gate + rearrange(bias, "h n -> () h n ()")

        gate = rearrange(gate, "b h n d -> b n (h d)")

        if exists(gate_res):
            gate = gate + gate_res

        return self.act(gate) * res


class gMLPBlock(nn.Module):
    def __init__(self, *, dim, dim_ff, seq_len, heads=1, attn_dim=None, causal=False, act=nn.Identity(), circulant_matrix=False):
        super().__init__()
        self.proj_in = nn.Sequential(nn.Linear(dim, dim_ff), nn.GELU())

        self.attn = Attention(dim, dim_ff // 2, attn_dim, causal) if exists(attn_dim) else None

        self.sgu = SpatialGatingUnit(dim_ff, seq_len, causal, act, heads, circulant_matrix=circulant_matrix)
        self.proj_out = nn.Linear(dim_ff // 2, dim)

    def forward(self, x):
        gate_res = self.attn(x) if exists(self.attn) else None
        x = self.proj_in(x)
        x = self.sgu(x, gate_res=gate_res)
        x = self.proj_out(x)
        return x


class gMLP(nn.Module):
    def __init__(
        self,
        *,
        # embedding dim (ie nb of possible words). Also the output shape
        output_dim,
        input_dim,  # dimension of the embedding (if None, input dim)
        depth,  # number of blocks
        seq_len,  # len of seq
        heads=1,  # keep 1 (multi-head otherwise, not in original paper)
        ff_mult=4,  # hidden sim of channel projetion is dim*ff_mult
        attn_dim=None,  # True for inference ?
        prob_survival=1.0,
        causal=False,
        circulant_matrix=False,  # use circulant weight matrix for linear
        # increase in parameters in respect to sequence length
        shift_tokens=0,
        # no activation for spatial gate (act) in original paper; author suggests nn.Tanh()
        act=nn.Identity()
    ):
        super().__init__()
        assert (input_dim % heads) == 0, "dimension must be divisible by number of heads"

        dim_ff = input_dim * ff_mult
        self.seq_len = seq_len
        self.prob_survival = prob_survival

        # self.to_embed = nn.Embedding(num_tokens, dim) if exists(num_tokens) else nn.Identity()

        token_shifts = tuple(range(0 if causal else -shift_tokens, shift_tokens + 1))
        self.layers = nn.ModuleList(
            [
                Residual(
                    PreNorm(
                        input_dim,
                        PreShiftTokens(
                            token_shifts,
                            gMLPBlock(
                                dim=input_dim,
                                heads=heads,
                                dim_ff=dim_ff,
                                seq_len=seq_len,
                                attn_dim=attn_dim,
                                causal=causal,
                                act=act,
                                circulant_matrix=circulant_matrix,
                            ),
                        ),
                    )
                )
                for i in range(depth)
            ]
        )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim)
            ) #if exists(output_dim) else nn.Identity()


    def forward(self, x):

        # x = self.to_embed(x)
        layers = self.layers if not self.training else dropout_layers(self.layers, self.prob_survival)
        out = nn.Sequential(*layers)(x)
        
        return self.to_logits(out)
        #return out


def padding(x, seq_len, device, central_ft=None, dist=None):

    # order neighbours from closer to further
    if dist is not None:
        _, ids = torch.sort(dist)
        x = x[ids]

    else:
        _, ids = torch.sort(x, -2)
        x = x[ids[:, -1]]

    # Add a first line corresponding to the central residue
    if central_ft is not None:
        x = torch.cat((x, torch.zeros((x.shape[0], central_ft.shape[-1] - x.shape[1]), device=device)), dim=1)
        x = torch.cat((central_ft.reshape(1, -1), x))

    if seq_len - x.shape[0] > 0:
        pad = torch.zeros((seq_len - x.shape[0], x.shape[-1])).to(device)
        x_pad = torch.cat((x, pad))

    else:  # if there are too many neighbours, the last ones are cut
        x_pad = x[:seq_len]

    x_pad = torch.unsqueeze(x_pad, 0)
    return x_pad
