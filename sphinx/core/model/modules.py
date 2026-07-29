import math

import torch
# from IPython import embed
from timm.layers import DropPath, Mlp
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


class MHSA(nn.Module):

    def __init__(self, dim, seq_len, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # sparse self-attn
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.k_sr = nn.Conv1d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, padding=0)
            self.v_sr = nn.Conv1d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, padding=0)

        # incremental settings
        self.incremental_infer = False
        # Note 1: k_cache and v_cache are the results before strided conv, and they are not used if sr_ratio == 1
        # Note 2: k_sr_cache and v_sr_cache are the results after strided conv, and they are updated every sr_ratio timesteps
        self.incremental_cache = {'idx': 0, 'k_cache': None, 'k_sr_cache': None, 'v_cache': None, 'v_sr_cache': None}
        self.kv_sr_cache_len = math.ceil(seq_len / sr_ratio) - 1  # cache length after strided conv, e.g., cache len 29 for seq_len = 478, sr_ratio = 16

    def forward(self, x):
        if self.incremental_infer:
            return self.incremental_forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if self.sr_ratio > 1:
            k = k.permute(0, 2, 1, 3).reshape(B, N, -1).permute(0, 2, 1)
            k = self.k_sr(k).permute(0, 2, 1).reshape(B, N // self.sr_ratio, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            v = v.permute(0, 2, 1, 3).reshape(B, N, -1).permute(0, 2, 1)
            v = self.v_sr(v).permute(0, 2, 1).reshape(B, N // self.sr_ratio, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # gen attention mask
        attn_mask = attn.new_ones(1, 1, attn.size(2), attn.size(2))  # [1,1,1024,1024]
        attn_mask = torch.triu(attn_mask, diagonal=0)
        if self.sr_ratio > 1:
            attn_mask = F.max_pool2d(attn_mask, kernel_size=(1, self.sr_ratio))
        attn = attn.masked_fill(attn_mask == 1, float('-inf')).softmax(dim=-1).nan_to_num()
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def incremental_forward(self, x):
        idx = self.incremental_cache['idx']
        N, T, C = x.shape  # N: symbols, T: time steps, C: channels
        assert T == 1
        qkv = self.qkv(x).reshape(N, 1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (N, num_heads, 1, C//num_heads)

        update_sr_cache = (idx + 1) % self.sr_ratio == 0  # update k_sr_cache and v_sr_cache at the last step of each sr block

        if self.sr_ratio > 1:
            k = k.permute(0, 2, 1, 3).reshape(N, 1, -1).permute(0, 2, 1)  # (N, C, 1)
            k_cache = self.incremental_cache['k_cache']
            if k_cache is None:
                k_cache = torch.zeros((N, C, self.sr_ratio - 1), device=k.device)
            elif k_cache is not None:
                pass
            else:
                raise ValueError("unexpected k_cache state")
            k_cache = torch.cat([k_cache, k], dim=-1)  # (N, C, sr_ratio)
            self.incremental_cache['k_cache'] = k_cache[:, :, 1:]  # update k_cache, (N, C, sr_ratio-1)
            # prepare for k_sr_cache update
            if update_sr_cache:
                k_sr_cache_update = self.k_sr(k_cache).permute(0, 2, 1).reshape(N, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (N, num_heads, 1, C//num_heads)

            v = v.permute(0, 2, 1, 3).reshape(N, T, -1).permute(0, 2, 1)  # (N, C, 1)
            v_cache = self.incremental_cache['v_cache']
            if v_cache is None:
                v_cache = torch.zeros((N, C, self.sr_ratio - 1), device=v.device)
            elif v_cache is not None:
                pass
            else:
                raise ValueError("unexpected v_cache state")
            v = torch.cat([v_cache, v], dim=-1)  # (N, C, sr_ratio)
            self.incremental_cache['v_cache'] = v[:, :, 1:]  # update v_cache, (N, C, sr_ratio-1)
            # prepare for v_sr_cache update
            if update_sr_cache:
                v_sr_cache_update = self.v_sr(v).permute(0, 2, 1).reshape(N, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (N, num_heads, 1, C//num_heads)
        elif self.sr_ratio == 1:
            # update k_sr_cache every step
            k_sr_cache_update = k
            v_sr_cache_update = v
        else:
            raise ValueError(f"sr_ratio should be positive, but got {self.sr_ratio}")

        k_sr_cache = self.incremental_cache['k_sr_cache']
        attn_last_inf = False
        if k_sr_cache is None:
            k = torch.zeros((N, self.num_heads, 1, C // self.num_heads), device=k.device)
            attn_last_inf = True  # set the last attention before softmax to be -inf
        elif k_sr_cache is not None:
            k = k_sr_cache  # (N, num_heads, t <= seq_len/sr_ratio - 1, C//num_heads)
        else:
            raise ValueError("unexpected k_sr_cache state")
        # update k_sr_cache
        if update_sr_cache:
            if k_sr_cache is not None:
                k_sr_cache = torch.cat([k_sr_cache, k_sr_cache_update], dim=2)
            elif k_sr_cache is None:
                k_sr_cache = k_sr_cache_update
            else:
                raise ValueError("unexpected k_sr_cache state")
            self.incremental_cache['k_sr_cache'] = k_sr_cache[:, :, -self.kv_sr_cache_len:, :]  # (N, num_heads, t <= seq_len/sr_ratio - 1, C//num_heads)

        v_sr_cache = self.incremental_cache['v_sr_cache']
        if v_sr_cache is None:
            v = torch.zeros((N, self.num_heads, 1, C // self.num_heads), device=v.device)
        elif v_sr_cache is not None:
            v = v_sr_cache  # (N, num_heads, t <= seq_len/sr_ratio - 1, C//num_heads)
        else:
            raise ValueError("unexpected v_sr_cache state")
        # update v_sr_cache
        if update_sr_cache:
            if v_sr_cache is not None:
                v_sr_cache = torch.cat([v_sr_cache, v_sr_cache_update], dim=2)
            elif v_sr_cache is None:
                v_sr_cache = v_sr_cache_update
            else:
                raise ValueError("unexpected v_sr_cache state")
            self.incremental_cache['v_sr_cache'] = v_sr_cache[:, :, -self.kv_sr_cache_len:, :]  # (N, num_heads, t <= seq_len/sr_ratio - 1, C//num_heads)

        # update idx
        self.incremental_cache['idx'] = (idx + 1) % self.sr_ratio

        # gen attention mask
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (N, num_heads, 1, t)
        if attn_last_inf:
            attn[..., -1] = float('-inf')
        attn = attn.softmax(dim=-1).nan_to_num()
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(N, 1, C)
        x = self.proj(x)

        x = self.proj_drop(x)
        return x

    def reset_eval(self, mode):
        if mode == 'eval':
            self.incremental_infer = False
        elif mode == 'incremental_eval':
            self.incremental_infer = True
        else:
            raise ValueError(f'mode should be eval or incremental_eval, but got {mode}')

    def reset_cache(self):
        self.incremental_cache = {'idx': 0, 'k_cache': None, 'k_sr_cache': None, 'v_cache': None, 'v_sr_cache': None}


class GraphMHSA(nn.Module):
    """Multi-Head Self-Attention in Graphs"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.incremental_infer = False

    def forward(self, x, adj):
        if self.incremental_infer:
            return self.incremental_forward(x, adj)
        # if not self.training:
        #     if self.incremental_infer:
        #         return self.incremental_forward(x, adj)
        #     else:
        #         return self.forward_infer(x, adj)

        B, T, N, C = x.shape
        BT = B * T
        x = x.reshape(BT, N, C)
        qkv = self.qkv(x).reshape(BT, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # q, k, v: (BT, num_heads, N, C//num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (BT, num_heads, N, N)
        attn = attn.reshape(B, T, self.num_heads, N, N)

        # apply adjacency matrix
        adj_mask = adj.reshape(B, 1, 1, N, N)
        attn = attn.masked_fill(adj_mask == 0, float('-inf')).softmax(dim=-1).nan_to_num()  # (B, T, num_heads, N, N)
        attn = self.attn_drop(attn)
        attn = attn.reshape(BT, self.num_heads, N, N)

        # attn.shape: (BT, num_heads, N, N)
        # v.shape: (BT, num_heads, N, C // num_heads)
        x = (attn @ v).transpose(1, 2).reshape(BT, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.reshape(B, T, N, C)
        return x

    def forward_infer(self, x, adj):
        B, T, N, C = x.shape
        assert B == 1
        x = x.reshape(T, N, C)
        # hard code: 10 min per forward
        step = 10
        t1 = 0
        while t1 < T:
            t2 = min(t1 + step, T)
            x_cur = x[t1:t2]
            qkv = self.qkv(x_cur).reshape(t2 - t1, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # q, k, v: (t2-t1, num_heads, N, C//num_heads)

            attn = (q @ k.transpose(-2, -1)) * self.scale  # (t2-t1, num_heads, N, N)
            attn = attn.reshape(t2 - t1, self.num_heads, N, N)

            # apply adjacency matrix
            attn = attn.masked_fill(adj == 0, float('-inf')).softmax(dim=-1).nan_to_num()  # (t2-t1, num_heads, N, N)
            attn = self.attn_drop(attn)
            attn = attn.reshape(t2 - t1, self.num_heads, N, N)

            # attn.shape: (t2-t1, num_heads, N, N)
            # v.shape: (t2-t1, num_heads, N, C // num_heads)
            x_cur = (attn @ v).transpose(1, 2).reshape(t2 - t1, N, C)
            x_cur = self.proj(x_cur)
            x_cur = self.proj_drop(x_cur)
            x[t1:t2] = x_cur
            t1 = t2

        x = x.reshape(B, T, N, C)
        return x

    def incremental_forward(self, x, adj):
        B, T, N, C = x.shape
        assert B == 1
        assert T == 1
        x = x.reshape(1, N, C)

        qkv = self.qkv(x).reshape(1, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # q, k, v: (1, num_heads, N, C//num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (1, num_heads, N, N)
        attn = attn.reshape(1, self.num_heads, N, N)

        # apply adjacency matrix
        attn = attn.masked_fill(adj == 0, float('-inf')).softmax(dim=-1).nan_to_num()  # (1, num_heads, N, N)
        attn = self.attn_drop(attn)
        attn = attn.reshape(1, self.num_heads, N, N)

        # attn.shape: (1, num_heads, N, N)
        # v.shape: (1, num_heads, N, C // num_heads)
        x = (attn @ v).transpose(1, 2).reshape(1, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.reshape(B, T, N, C)
        return x

    def reset_eval(self, mode):
        if mode == 'eval':
            self.incremental_infer = False
        elif mode == 'incremental_eval':
            self.incremental_infer = True
        else:
            raise ValueError(f'mode should be eval or incremental_eval, but got {mode}')

    def reset_cache(self):
        # no incremental cache in GraphMHSA
        pass


class GraphMHSA_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, adj_num=1):
        super().__init__()

        self.norm1 = nn.ModuleList()
        self.attn = nn.ModuleList()
        self.adj_num = adj_num
        for _ in range(adj_num):
            self.norm1.append(norm_layer(dim))
            self.attn.append(GraphMHSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop))
        if drop_path > 0.:
            self.drop_path = DropPath(drop_path)
        elif drop_path <= 0.:
            self.drop_path = nn.Identity()
        else:
            raise ValueError(f"unexpected drop_path: {drop_path}")

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, adj):
        for i in range(self.adj_num):
            x = x + self.drop_path(self.attn[i](self.norm1[i](x), adj[:, i:i + 1, :, :]))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def incremental_forward(self, x):
        idx = self.incremental_cache['idx']
        N, T, C = x.shape  # N: symbols, T: time steps, C: channels
        assert T == 1
        qkv = self.qkv(x).reshape(N, 1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (N, num_heads, 1, C//num_heads)

        update_sr_cache = (idx + 1) % self.sr_ratio == 0  # update k_sr_cache and v_sr_cache at the last step of each sr block

        if self.sr_ratio > 1:
            k = k.permute(0, 2, 1, 3).reshape(N, 1, -1).permute(0, 2, 1)  # (N, C, 1)
            k_cache = self.incremental_cache['k_cache']
            if k_cache is None:
                k_cache = torch.zeros((N, C, self.sr_ratio - 1), device=k.device)
            elif k_cache is not None:
                pass
            else:
                raise ValueError("unexpected k_cache state")
            k_cache = torch.cat([k_cache, k], dim=-1)  # (N, C, sr_ratio)
            self.incremental_cache['k_cache'] = k_cache[:, :, 1:]  # update k_cache, (N, C, sr_ratio-1)
            # prepare for k_sr_cache update
            if update_sr_cache:
                k_sr_cache_update = self.k_sr(k_cache).permute(0, 2, 1).reshape(N, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (N, num_heads, 1, C//num_heads)

            v = v.permute(0, 2, 1, 3).reshape(N, T, -1).permute(0, 2, 1)  # (N, C, 1)
            v_cache = self.incremental_cache['v_cache']
            if v_cache is None:
                v_cache = torch.zeros((N, C, self.sr_ratio - 1), device=v.device)
            elif v_cache is not None:
                pass
            else:
                raise ValueError("unexpected v_cache state")
            v = torch.cat([v_cache, v], dim=-1)  # (N, C, sr_ratio)
            self.incremental_cache['v_cache'] = v[:, :, 1:]  # update v_cache, (N, C, sr_ratio-1)
            # prepare for v_sr_cache update
            if update_sr_cache:
                v_sr_cache_update = self.v_sr(v).permute(0, 2, 1).reshape(N, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (N, num_heads, 1, C//num_heads)
        elif self.sr_ratio == 1:
            # update k_sr_cache every step
            k_sr_cache_update = k
            v_sr_cache_update = v
        else:
            raise ValueError(f"sr_ratio should be positive, but got {self.sr_ratio}")

        k_sr_cache = self.incremental_cache['k_sr_cache']
        attn_last_inf = False
        if k_sr_cache is None:
            k = torch.zeros((N, self.num_heads, 1, C // self.num_heads), device=k.device)
            attn_last_inf = True  # set the last attention before softmax to be -inf
        elif k_sr_cache is not None:
            k = k_sr_cache  # (N, num_heads, t <= seq_len/sr_ratio - 1, C//num_heads)
        else:
            raise ValueError("unexpected k_sr_cache state")
        # update k_sr_cache
        if update_sr_cache:
            if k_sr_cache is not None:
                k_sr_cache = torch.cat([k_sr_cache, k_sr_cache_update], dim=2)
            elif k_sr_cache is None:
                k_sr_cache = k_sr_cache_update
            else:
                raise ValueError("unexpected k_sr_cache state")
            self.incremental_cache['k_sr_cache'] = k_sr_cache[:, :, -self.kv_sr_cache_len:, :]  # (N, num_heads, t <= seq_len/sr_ratio - 1, C//num_heads)

        v_sr_cache = self.incremental_cache['v_sr_cache']
        if v_sr_cache is None:
            v = torch.zeros((N, self.num_heads, 1, C // self.num_heads), device=v.device)
        elif v_sr_cache is not None:
            v = v_sr_cache  # (N, num_heads, t <= seq_len/sr_ratio - 1, C//num_heads)
        else:
            raise ValueError("unexpected v_sr_cache state")
        # update v_sr_cache
        if update_sr_cache:
            if v_sr_cache is not None:
                v_sr_cache = torch.cat([v_sr_cache, v_sr_cache_update], dim=2)
            elif v_sr_cache is None:
                v_sr_cache = v_sr_cache_update
            else:
                raise ValueError("unexpected v_sr_cache state")
            self.incremental_cache['v_sr_cache'] = v_sr_cache[:, :, -self.kv_sr_cache_len:, :]  # (N, num_heads, t <= seq_len/sr_ratio - 1, C//num_heads)

        # update idx
        self.incremental_cache['idx'] = (idx + 1) % self.sr_ratio

        # gen attention mask
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (N, num_heads, 1, t)
        if attn_last_inf:
            attn[..., -1] = float('-inf')
        attn = attn.softmax(dim=-1).nan_to_num()
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(N, 1, C)
        x = self.proj(x)

        x = self.proj_drop(x)
        return x

    def reset_eval(self, mode):
        if mode == 'eval':
            self.incremental_infer = False
        elif mode == 'incremental_eval':
            self.incremental_infer = True
        else:
            raise ValueError(f'mode should be eval or incremental_eval, but got {mode}')

    def reset_cache(self):
        self.incremental_cache = {'idx': 0, 'k_cache': None, 'k_sr_cache': None, 'v_cache': None, 'v_sr_cache': None}


class MHSA_Block(nn.Module):

    def __init__(self, dim, num_heads, seq_len, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = MHSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, seq_len=seq_len)
        if drop_path > 0.:
            self.drop_path = DropPath(drop_path)
        elif drop_path <= 0.:
            self.drop_path = nn.Identity()
        else:
            raise ValueError(f"unexpected drop_path: {drop_path}")

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


if __name__ == '__main__':
    # a test case for MHSA to test incremental_forward()
    N = 20
    T = 717
    C = 16

    torch.manual_seed(0)
    mhsa = MHSA(C, num_heads=2, qkv_bias=True, sr_ratio=16, seq_len=717)
    mhsa.eval()

    # incremental
    x_in = torch.randn(N, T, C)
    mhsa.reset_eval('incremental_eval')
    x_out = []
    for i in range(T):
        x = x_in[:, i:i + 1, :]
        x = mhsa(x)
        x_out.append(x)
    x_incremental = torch.cat(x_out, dim=1)

    # regular
    mhsa.reset_eval('eval')
    x_regular = mhsa(x_in)
    print((x_regular == x_incremental).all())
    print((x_regular - x_incremental).abs().max())
    print(f'identical ratio: {(x_regular == x_incremental).sum() / x_regular.numel()}')

    x_diff = (x_incremental - x_regular).abs()
    print(f'mean diff: {x_diff.mean()}, max diff: {x_diff.max()}')
    print(x_incremental[0, -1])
    print(x_regular[0, -1])
