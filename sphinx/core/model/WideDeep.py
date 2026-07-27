import copy
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.checkpoint import checkpoint

from sphinx.core.model.modules import GraphMHSA_Block, MHSA_Block


class CausalConv1d(nn.Module):
    """ Causal conv1d without changing the seqlen.

    Inputs:
        (B,C,T)

    Returns:
        (B,C,T)
    """

    def __init__(self, in_channel, out_channel, kernel_size, dilation, groups=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        # self.conv = weight_norm(nn.Conv1d(in_channel, out_channel, kernel_size, padding=self.padding, dilation=dilation, groups=groups))
        self.conv = weight_norm(nn.Conv1d(in_channel, out_channel, kernel_size, padding=self.padding, dilation=dilation, groups=groups))
        self.conv.weight.data.normal_(0, 0.01)
        self.incremental_infer = False
        self.incremental_cache = None

    def forward(self, x):
        if self.incremental_infer:
            return self.incremental_forward(x)
        elif not self.incremental_infer:
            x = self.conv(x)
            return x[..., :x.shape[-1] - self.padding].contiguous()
        else:
            raise ValueError(f"unexpected incremental_infer state: {self.incremental_infer}")

    def incremental_forward(self, x):
        if self.incremental_cache is None:
            self.incremental_cache = torch.zeros_like(x, device=x.device).repeat(1, 1, self.padding)
        x = torch.cat([self.incremental_cache, x], dim=-1)
        self.incremental_cache = x[:, :, -self.padding:]
        x = self.conv(x)
        return x[..., self.padding:x.shape[-1] - self.padding].contiguous()

    def reset_eval(self, mode):
        if mode == 'eval':
            self.incremental_infer = False
        elif mode == 'incremental_eval':
            self.incremental_infer = True
        else:
            raise ValueError(f'mode should be eval or incremental_eval, but got {mode}')

    def reset_cache(self):
        self.incremental_cache = None


class TemporalBlock(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.3, repeat=1, residual=True):
        super(TemporalBlock, self).__init__()
        self.residual = residual
        net = []
        for _ in range(repeat):
            net += [CausalConv1d(n_inputs, n_outputs, kernel_size, dilation=dilation), nn.ReLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*net)
        if n_inputs != n_outputs:
            self.downsample = nn.Conv1d(n_inputs, n_outputs, 1)
        elif n_inputs == n_outputs:
            self.downsample = None
        else:
            raise ValueError(f"unexpected channel state: {n_inputs}, {n_outputs}")
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
        elif self.downsample is None:
            pass
        else:
            raise ValueError("unexpected downsample state")

    def forward(self, x):
        out = self.net(x)
        if self.residual:
            if self.downsample is not None:
                res = self.downsample(x)
            elif self.downsample is None:
                res = x
            else:
                raise ValueError("unexpected downsample state")
            return F.relu(out + res)
        elif not self.residual:
            return F.relu(out)
        else:
            raise ValueError(f"unexpected residual state: {self.residual}")


class TemporalConvNet(nn.Module):

    def __init__(self, num_inputs, num_channels, kernel_size, dilation_size, dropout=0.3, residual=True, repeat=1):
        super(TemporalConvNet, self).__init__()
        self.residual = residual
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            if i == 0:
                in_channels = num_inputs
            elif i != 0:
                in_channels = num_channels[i - 1]
            else:
                raise ValueError(f"unexpected level index: {i}")
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size[i], dilation=dilation_size[i], dropout=dropout, repeat=repeat, residual=residual)]
        layers.append(nn.Conv1d(num_channels[-1], num_channels[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GRU(nn.GRU):
    """GRU with incremental inference support"""

    def __init__(self, input_size, hidden_size, batch_first):
        super().__init__(input_size, hidden_size, batch_first=batch_first)
        self.incremental_infer = False
        self.incremental_cache = None

    def forward(self, input, hx=None):
        if self.incremental_infer:
            return self.incremental_forward(input)
        return super().forward(input, hx=hx)

    def incremental_forward(self, x):
        if self.incremental_cache is None:
            xn, hn = super().forward(x)
        elif self.incremental_cache is not None:
            xn, hn = super().forward(x, self.incremental_cache)
        else:
            raise ValueError("unexpected incremental cache state")
        self.incremental_cache = hn
        return xn, hn

    def reset_eval(self, mode):
        if mode == 'eval':
            self.incremental_infer = False
        elif mode == 'incremental_eval':
            self.incremental_infer = True
        else:
            raise ValueError(f'mode should be eval or incremental_eval, but got {mode}')

    def reset_cache(self):
        self.incremental_cache = None


class LSTM(nn.LSTM):
    """LSTM with incremental inference support"""

    def __init__(self, input_size, hidden_size, batch_first):
        super().__init__(input_size, hidden_size, batch_first=batch_first)
        self.incremental_infer = False
        self.incremental_cache = None

    def forward(self, input, hx=None):
        if self.incremental_infer:
            return self.incremental_forward(input)
        return super().forward(input, hx=hx)

    def incremental_forward(self, x):
        if self.incremental_cache is None:
            xn, hn = super().forward(x)
        elif self.incremental_cache is not None:
            xn, hn = super().forward(x, self.incremental_cache)
        else:
            raise ValueError("unexpected incremental cache state")
        self.incremental_cache = hn
        return xn, hn

    def reset_eval(self, mode):
        if mode == 'eval':
            self.incremental_infer = False
        elif mode == 'incremental_eval':
            self.incremental_infer = True
        else:
            raise ValueError(f'mode should be eval or incremental_eval, but got {mode}')

    def reset_cache(self):
        self.incremental_cache = None


class RNN_MHSA_Graph(nn.Module):

    def __init__(
            self,
            channel,
            layers=4,
            rnn_type="lstm",
            mhsa_params=dict(num_heads=2, qkv_bias=True, drop=0.1, drop_path=0.3, attn_drop=0.1, sr_ratio=16),
            graph_params=dict(num_heads=2, qkv_bias=True, drop=0.1, drop_path=0.3, attn_drop=0.1),
    ):
        super().__init__()
        assert rnn_type in ['lstm', 'gru']
        rnn_params = dict(input_size=channel, hidden_size=channel, batch_first=True)
        self.mhsa_block = nn.ModuleList()
        self.rnn_block = nn.ModuleList()
        self.graph_block = nn.ModuleList()
        for _ in range(layers):
            if rnn_type == 'lstm':
                self.rnn_block.append(LSTM(**rnn_params))
            elif rnn_type == 'gru':
                self.rnn_block.append(GRU(**rnn_params))
            else:
                raise ValueError('rnn_type should be lstm, gru')
            self.mhsa_block.append(MHSA_Block(channel, **mhsa_params))
            self.graph_block.append(GraphMHSA_Block(channel, **graph_params))

    def forward(self, x, adj, ckpt=False):
        # x.shape: (B, N, C, T)
        B, N, C, T = x.shape
        for i, mhsa_block in enumerate(self.mhsa_block):
            x = x.contiguous().view(B * N, C, T).permute(0, 2, 1)  # (B, T, C)
            xn, _ = self.rnn_block[i](x)
            x = x + xn
            if self.training and ckpt:
                x = checkpoint(mhsa_block, x, use_reentrant=True)
            elif not self.training or not ckpt:
                x = mhsa_block(x)
            else:
                raise ValueError(f"unexpected checkpoint state: training={self.training}, ckpt={ckpt}")
            x = x.permute(0, 2, 1)
            x = x.contiguous().view(B, N, C, T)
            x = x.permute(0, 3, 1, 2)  # (B, T, N, C)
            x = self.graph_block[i](x, adj)  # residual is in the graph block
            x = x.permute(0, 2, 3, 1)  # (B, N, C, T)
        return x


class WideDeepGATProb(nn.Module):
    """(rnn + trans + graph attn) in each block"""

    def __init__(
        self,
        wide_params,
        deep_params,
        in_channels,
        mid_channels,
        out_channels,
        seq_len,
        dropout,
        wide_gradient_ckpt,
        deep_gradient_ckpt,
        receptive_field,
    ):
        super(WideDeepGATProb, self).__init__()
        self.droppad = False
        self.wide_gradient_ckpt = wide_gradient_ckpt
        self.deep_gradient_ckpt = deep_gradient_ckpt
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.receptive_field = receptive_field
        self.seq_len = seq_len
        self.eval_mode = 'eval'

        net_list = []
        for param in wide_params:
            if param == "input":
                net_list.append(nn.Identity())
            elif param != "input":
                net_list.append(
                    TemporalConvNet(in_channels, [mid_channels] * len(param["dilation_size"]), [param["kernel_size"]] * len(param["dilation_size"]), param["dilation_size"], dropout, residual=True))
            else:
                raise ValueError(f"unexpected wide param: {param}")
        self.net = nn.ModuleList(net_list)

        self.use_pos = False
        channel = mid_channels * len(net_list)
        deep_params['params']['channel'] = channel
        deep_params['params']['mhsa_params']['seq_len'] = seq_len
        self.deep = RNN_MHSA_Graph(**deep_params['params'])

        self.last_conv = nn.Sequential(*[nn.ReLU(), nn.Conv1d(channel, channel // 2, 1), nn.ReLU(), nn.Dropout(dropout), nn.Conv1d(channel // 2, self.out_channels, 1)])
        self.last_conv_prob = nn.Sequential(*[nn.ReLU(), nn.Conv1d(channel, channel // 2, 1), nn.ReLU(), nn.Dropout(dropout), nn.Conv1d(channel // 2, self.out_channels, 1)])

    def forward(self, x, adj=None, **kwargs):
        # x: (B, N, C, T)
        # adj: (B, adj_num, N, N)
        B, N, C, T = x.size()  # batch size, symbol num, channel, length
        assert adj is None
        adj = torch.ones(B, 2, N, N, dtype=torch.int8).to(x.device)
        x = x.view(B * N, C, T).contiguous()

        skips = []
        for f in self.net:
            if self.training and self.wide_gradient_ckpt:
                z = checkpoint(f, x)
            elif not self.training or not self.wide_gradient_ckpt:
                z = f(x)
            else:
                raise ValueError(f"unexpected checkpoint state: training={self.training}, ckpt={self.wide_gradient_ckpt}")
            skips.append(z)
        x = torch.cat(skips, dim=1)

        x = x.contiguous().view(B, N, -1, T)
        x = self.deep(x, adj, self.deep_gradient_ckpt)

        x_tmp = x.contiguous().view(B * N, -1, T)
        x = self.last_conv(x_tmp)
        x_prob = self.last_conv_prob(x_tmp)
        x = x.view(B, N, self.out_channels, -1)
        x_prob = x_prob.view(B, N, self.out_channels, -1)

        return x, x_prob

    def reset_eval_model(self, mode):
        if mode in ['eval', 'incremental_eval']:
            for module in self.modules():
                if hasattr(module, 'reset_eval'):
                    module.reset_eval(mode)
            self.eval_mode = mode
            return self.eval()
        else:
            raise ValueError(f'mode should be eval or incremental_eval, but got {mode}')

    def reset_cache_model(self):
        for module in self.modules():
            if hasattr(module, 'reset_cache'):
                module.reset_cache()
        return self
