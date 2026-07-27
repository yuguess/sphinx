import torch
from IPython import embed
from torch import nn
from torch.nn import functional as F

# from torch.nn import L1Loss, MSELoss, SmoothL1Loss

EPS = 1e-5


def replace_dim_with_mean(tensor, target_dim):
    """
    沿指定维度计算均值，并将该维度所有元素替换为均值
    :param tensor: 输入多维Tensor
    :param target_dim: 目标维度（int类型，需在tensor的维度范围内）
    :return: 替换后的Tensor
    """
    # 1. 沿目标维度计算均值，keepdim=True保持原维度结构（避免降维导致广播失败）
    dim_mean = tensor.mean(dim=target_dim, keepdim=True)

    # 2. 用均值Tensor替换原Tensor的目标维度（利用广播机制）
    tensor_replaced = tensor.clone()  # 克隆原Tensor避免修改原始数据
    tensor_replaced = dim_mean.expand_as(tensor_replaced)  # 扩展均值维度以匹配原Tensor
    # 或简化为：tensor_replaced = dim_mean.broadcast_to(tensor_replaced.shape)

    return tensor_replaced


def weighted_mean(x, w, dim=0, keepdim=False):
    return (w * x).sum(dim=dim, keepdim=keepdim) / w.sum(dim=dim, keepdim=keepdim)


def weighted_std(x, w, dim=0, keepdim=False):
    x_mean = weighted_mean(x, w, dim=dim, keepdim=True)
    return torch.sqrt(((x - x_mean)**2 * w).sum(dim=dim, keepdim=keepdim) / w.sum(dim=dim, keepdim=keepdim))


class PairwiseLoss(nn.Module):

    def __init__(self, channels=None, key_prefix=None, dims=float("nan")):
        """
        Args:
            dims: 要进行配对的维度，可以是单个整数或维度列表/元组
        """
        super().__init__()
        self.dims = [dims]
        if isinstance(dims, (list, tuple)):
            self.dims = dims
        self.channels = channels
        self.key_prefix = key_prefix

    def forward(self, x_dict, y_dict, _loss_weight):

        pred = x_dict[self.key_prefix + "pred"][:, :, self.channels, :]
        label = y_dict[self.key_prefix + "label"][:, :, self.channels, :]
        B, N, C, T = pred.shape

        valid = ~label.isnan()
        pred = pred.masked_fill(~valid, 0)
        label = label.masked_fill(~valid, 0)

        # --------------------------
        # 1. 将指定维度展平为单一维度
        # --------------------------
        # 构造维度重排顺序：先非指定维度，后指定维度

        non_dims = [d for d in range(pred.dim()) if d not in self.dims]
        permute_order = non_dims + self.dims

        # 重排并展平指定维度
        pred_perm = pred.permute(*permute_order).contiguous()
        label_perm = label.permute(*permute_order).contiguous()

        flat_shape = pred_perm.shape[:len(non_dims)] + (-1,)
        pred_flat = pred_perm.view(flat_shape)
        label_flat = label_perm.view(flat_shape)

        # --------------------------
        # 2. 在展平维度上随机配对
        # --------------------------
        pair_dim = -1  # 展平后的维度位于最后
        size = pred_flat.size(pair_dim)
        size_even = size - (size % 2)  # 确保样本数为偶数

        if size_even < 2:
            raise ValueError("Insufficient number of samples")
            return torch.tensor(0.0, device=pred.device)

        # 生成随机索引并两两分组
        indices = torch.randperm(size, device=pred.device)[:size_even]
        idx1, idx2 = indices[0::2], indices[1::2]

        # --------------------------
        # 3. 计算配对差值与 Loss
        # --------------------------
        def get_pairs(x, idx):
            slices = [slice(None)] * x.dim()
            slices[pair_dim] = idx
            return x[tuple(slices)]

        # 取出配对样本
        pred1, pred2 = get_pairs(pred_flat, idx1), get_pairs(pred_flat, idx2)
        label1, label2 = get_pairs(label_flat, idx1), get_pairs(label_flat, idx2)

        # MSE Loss
        # return F.mse_loss(pred1 - pred2, label1 - label2)

        # margin rank loss
        return F.margin_ranking_loss(pred1, pred2, torch.sign(label1 - label2))


class MSELoss(nn.Module):

    def __init__(self, reduction=None, channels=None, key_prefix=None, diff=None, mean_dim=None):
        super().__init__()
        self.reduction = reduction
        assert reduction in ['mean', 'weight_mean', "mae", "var"]
        assert isinstance(channels, list)
        assert isinstance(key_prefix, str)
        self.channel = channels
        self.key_prefix = key_prefix
        self.label_key = f"{key_prefix}label"
        self.pred_key = f"{key_prefix}pred"
        self.diff = diff
        self.mean_dim = mean_dim

    def forward(
        self,
        x_dict,
        y_dict,
        loss_weight,
        # group_valid,
    ):
        pred = x_dict[self.pred_key][:, :, self.channel, :]
        label = y_dict[self.label_key][:, :, self.channel, :]
        B, N, C, T = pred.shape

        valid = ~label.isnan()

        # valid[~group_valid] = False

        pred = pred.masked_fill(~valid, 0)
        label = label.masked_fill(~valid, 0)
        if self.reduction == "mean":
            # valid = valid.flatten()
            # pred = pred.flatten()[valid]
            # label = label.flatten()[valid]
            # return ((pred - label)**2).mean()
            pred = pred.flatten()
            label = label.flatten()
            return ((pred - label)**2).mean()
        elif self.reduction == "weight_mean":
            pred = pred.flatten()
            label = label.flatten()
            loss_weight = loss_weight.flatten()
            return (((pred - label)**2) * loss_weight).sum() / loss_weight.sum()
        elif self.reduction == "mae":
            pred = pred.flatten()
            label = label.flatten()
            return (torch.abs(pred - label)).mean()
        elif self.reduction == "var":
            pred = pred.flatten()
            label = label.flatten()
            return (pred - label).var()
        else:
            raise NotImplementedError("Reduction type not implemented")


class MeanShiftLoss(nn.Module):

    def __init__(self, reduction=None, channels=None, dim=None, key_prefix=None):
        super().__init__()
        assert reduction in ['mean', 'dim_mean', 'skew', 'dim_mean_corr', 'batch_mean']
        assert isinstance(channels, list)
        assert isinstance(key_prefix, str)
        self.channel = channels
        self.key_prefix = key_prefix
        self.label_key = f"{key_prefix}label"
        self.pred_key = f"{key_prefix}pred"
        self.reduction = reduction
        self.dim = dim

    def forward(
        self,
        x_dict,
        y_dict,
        _loss_weight,
        # group_valid,
    ):
        pred = x_dict[self.pred_key][:, :, self.channel, :]
        label = y_dict[self.label_key][:, :, self.channel, :]

        B, N, C, T = pred.shape

        valid = ~label.isnan()
        # valid[~group_valid] = False
        pred = pred.masked_fill(~valid, 0)
        label = label.masked_fill(~valid, 0)

        if self.reduction == 'mean':
            # valid = valid.flatten()
            # pred = pred.flatten()[valid]
            # label = label.flatten()[valid]
            pred = pred.flatten()
            label = label.flatten()
            return (pred.mean() / pred.std() - label.mean() / label.std())**2
        elif self.reduction == 'skew':

            def skew(x):
                return ((x - x.mean()) / x.std()).pow(3).mean()

            pred = pred.flatten()
            label = label.flatten()
            return (skew(pred) - skew(label))**2
        elif self.reduction == 'dim_mean':
            pred_mean = pred.mean(dim=self.dim, keepdim=True).flatten()
            label_mean = label.mean(dim=self.dim, keepdim=True).flatten()
            # pred = pred / pred.std()
            # label = label / label.std()
            # return ((pred - label)**2).mean()
            return ((pred_mean / pred.std() - label_mean / label.std())).pow(2).mean()
        elif self.reduction == 'dim_mean_corr':
            pred_mean = pred.mean(dim=self.dim, keepdim=True).flatten()
            label_mean = label.mean(dim=self.dim, keepdim=True).flatten()
            norm_pred = pred / pred.std()
            norm_label = label / label.std()
            return 1 - global_corr(norm_pred, norm_label, None, None)
        elif self.reduction == 'batch_mean':
            pred = pred.reshape(B, -1)
            label = label.reshape(B, -1)
            pred_mean = pred.mean(dim=-1)
            label_mean = label.mean(dim=-1)
            pred_std = pred.std(dim=-1)
            label_std = label.std(dim=-1)
            return (pred_mean / pred_std - label_mean / label_std).pow(2).mean()
            # return (pred.mean() / pred.std() - label.mean() / label.std())**2
        else:
            raise NotImplementedError("Reduction type not implemented")


class BCELoss(nn.Module):

    def __init__(
        self,
        channels=None,
        key_prefix=None,
        weight_type=None,
    ):
        super().__init__()
        assert isinstance(channels, list)
        assert isinstance(key_prefix, str)
        self.channel = channels
        self.key_prefix = key_prefix
        self.label_key = f"{key_prefix}label"
        self.pred_key = f"{key_prefix}pred"
        assert weight_type in ['eq', 'label', 'label_and_weight']
        self.weight_type = weight_type

    def forward(
        self,
        x_dict,
        y_dict,
        loss_weight,
        # group_valid,
    ):
        pred = x_dict[self.pred_key][:, :, self.channel, :]
        label = y_dict[self.label_key][:, :, self.channel, :]
        B, N, C, T = pred.shape

        valid = ~label.isnan()
        # valid[~group_valid] = False
        pred = pred.masked_fill(~valid, 0)
        label = label.masked_fill(~valid, 0)
        # 每个 batch 是一样重要的，所以先在 batch 维度上归一化
        loss_weight = loss_weight.masked_fill(~valid, 0)
        # tmp = loss_weight.reshape(B, -1)
        # tmp = tmp.sum(dim=1, keepdim=False)
        # loss_weight = loss_weight / tmp.view(B, 1, 1, 1)

        valid = valid.flatten()
        pred = pred.flatten()
        label = label.flatten()
        loss_weight = loss_weight.flatten()

        bce_pred = pred
        bce_label = label.ge(0).float()
        if self.weight_type == 'eq':
            bce_weight = torch.ones_like(bce_label)
        elif self.weight_type == 'label':
            bce_weight = torch.abs(label)
        elif self.weight_type == 'label_and_weight':
            bce_weight = torch.abs(label) * loss_weight
            # bce_weight = torch.abs(label) * (loss_weight + loss_weight.mean())
        else:
            raise NotImplementedError
        bce_weight = bce_weight / bce_weight.sum()
        assert torch.isfinite(bce_weight).all()
        loss = (F.binary_cross_entropy_with_logits(bce_pred, bce_label, reduction="none") * bce_weight).sum()

        return loss


def section_corr_old(
    x,
    y,
    dim,
):
    # B, N, C, T = x.shape
    N = x.size(1)
    org_x = x
    org_y = y
    invalid = torch.isnan(y)
    valid = ~invalid
    valid_sum_1 = valid.sum(dim=dim, keepdim=True)
    valid_sum_1 = valid_sum_1.masked_fill(valid_sum_1 == 0, 1)
    y = y.masked_fill(invalid, 0)
    x = x.masked_fill(invalid, 0)
    x = x - x.sum(dim=dim, keepdim=True) / valid_sum_1
    y = y - y.sum(dim=dim, keepdim=True) / valid_sum_1
    x = x.masked_fill(invalid, 0)
    y = y.masked_fill(invalid, 0)

    stdx = ((x**2).sum(dim=dim, keepdim=True) / valid_sum_1).sqrt()
    stdy = ((y**2).sum(dim=dim, keepdim=True) / valid_sum_1).sqrt()
    stdx = stdx.masked_fill(stdx < EPS, EPS)
    stdy = stdy.masked_fill(stdy < EPS, EPS)
    x = x / stdx
    y = y / stdy
    # x = x / (stdx + EPS)
    # y = y / (stdy + EPS)

    corr = ((x * y).sum(dim=dim, keepdim=True) / valid_sum_1).flatten()
    # corr_valid = (valid_sum_1.flatten() > (N // 2)) & (stdy.flatten() > EPS) & (torch.isfinite(corr))
    corr_valid = (stdy.flatten() > EPS) & (torch.isfinite(corr))
    # if corr_valid.sum() / len(corr_valid) < 0.5:
    #     print(f"valid mean: {corr_valid.sum() / len(corr_valid)}")
    #     return 0
    corr = corr[corr_valid]
    return corr.mean()


def section_corr(
    x,
    y,
    dim,
    ra,
    _loss_weight,
    # group_valid,
):
    B, N, C, T = x.shape
    # N = x.size(1)
    org_x = x
    org_y = y
    invalid = torch.isnan(y)
    valid = ~invalid
    # valid[~group_valid] = False
    y = y.masked_fill(invalid, 0)
    x = x.masked_fill(invalid, 0)

    x = x - x.mean(dim=dim, keepdim=True)
    y = y - y.mean(dim=dim, keepdim=True)
    stdx = x.std(dim=dim, keepdim=True)
    stdy = y.std(dim=dim, keepdim=True)

    x = x / (stdx + EPS)
    y = y / (stdy + EPS)

    # corr = ((x * y).mean(dim=dim, keepdim=True)).view(B, -1).mean(dim=1)
    corr = ((x * y).mean(dim=dim, keepdim=True)).flatten()
    corr = corr[torch.isfinite(corr)]
    if corr.numel() == 0:
        return x.new_tensor(0.0)
    if ra is not None:
        corr = corr.mean() - ra * corr.std(unbiased=False)
    elif ra is None:
        corr = corr.mean()
    else:
        raise ValueError(f"unexpected ra state: {ra}")
    return corr


# def global_corr(a, b):
#     a = a.flatten()
#     b = b.flatten()
#     valid = ~b.isnan()
#     # a = a.masked_fill(~valid, 0)
#     # b = b.masked_fill(~valid, 0)
#     # valid = torch.ones_like(valid.flatten(), dtype=torch.bool)
#     a = a[valid]
#     b = b[valid]
#     corr = ((a * b).mean() - a.mean() * b.mean()) / (a.std() * b.std())
#     return corr


def global_corr(
    a,
    b,
    _loss_weight,
    ra,
    # group_valid,
):
    B, N, C, T = a.shape
    valid = ~b.isnan()
    # valid[~group_valid] = False
    a = a.masked_fill(~valid, 0)
    b = b.masked_fill(~valid, 0)

    a = a - a.flatten().mean()
    b = b - b.flatten().mean()
    a_std = a.flatten().std()
    b_std = b.flatten().std()
    if (not torch.isfinite(a_std)) or (not torch.isfinite(b_std)) or a_std < EPS or b_std < EPS:
        return a.new_tensor(0.0)
    a = a / a_std
    b = b / b_std

    if ra is not None:
        # corr = (a * b).reshape(B, -1).mean(dim=1)
        # corr = corr.mean() - ra * corr.std()
        corr = (a * b).mean(dim=1).flatten()
        corr = corr[torch.isfinite(corr)]
        if corr.numel() == 0:
            return a.new_tensor(0.0)
        corr = corr.mean() - ra * corr.std(unbiased=False)
    elif ra is None:
        corr = (a * b).flatten().mean()
    else:
        raise ValueError(f"unexpected ra state: {ra}")
    return corr


def global_label_corr(
    x,
    y,
    dim,
    _loss_weight,
):
    invalid = torch.isnan(y)
    y = y.masked_fill(invalid, 0)
    x = x.masked_fill(invalid, 0)

    x = x - x.mean(dim=dim, keepdim=True)
    x_std = x.std(dim=dim, keepdim=True)
    x = x / (x_std + EPS)

    y = y - y.flatten().mean()
    y_std = y.flatten().std()
    if (not torch.isfinite(y_std)) or y_std < EPS:
        return x.new_tensor(0.0)
    y = y / y_std

    corr = (x * y).flatten()
    corr = corr[torch.isfinite(corr)]
    if corr.numel() == 0:
        return x.new_tensor(0.0)
    return corr.mean()


# def global_corr(a, b, _loss_weight, ra):
#     assert ra is None
#     B, N, C, T = a.shape
#     valid = ~b.isnan()
#     a = a.masked_fill(~valid, 0)
#     b = b.masked_fill(~valid, 0)
#     # valid = valid.flatten()
#     # a = a.flatten()[valid]
#     # b = b.flatten()[valid]
#     a = (a - a.mean()) / a.std()
#     b = (b - b.mean()) / b.std()
#     corr = (a * b).mean()
#     return corr


def glosec_corr(
    x,
    y,
    dim,
):
    # B, N, C, T = x.shape
    invalid = torch.isnan(y)
    valid = ~invalid
    valid_sum_1 = valid.sum(dim=dim, keepdim=True)
    valid_sum_1 = valid_sum_1.masked_fill(valid_sum_1 == 0, 1)
    y = y.masked_fill(invalid, 0)
    x = x.masked_fill(invalid, 0)
    x = x - x.sum(dim=dim, keepdim=True) / valid_sum_1
    y = y - y.sum(dim=dim, keepdim=True) / valid_sum_1
    x = x.masked_fill(invalid, 0)
    y = y.masked_fill(invalid, 0)

    valid = valid.flatten()
    x = x.flatten()[valid]
    y = y.flatten()[valid]
    stdx = x.std()
    stdy = y.std()
    x = x / stdx
    y = y / stdy

    corr = (x * y).mean()
    return corr


class CORRLoss(nn.Module):

    def __init__(self, reduction='mean', channels=None, key_prefix=None, dim=None, ra=None):
        super().__init__()
        self.reduction = reduction
        self.channels = channels
        self.key_prefix = key_prefix
        self.label_key = f"{key_prefix}label"
        self.pred_key = f"{key_prefix}pred"
        self.ra = ra
        if reduction in ['section', 'global_label_corr', 'channel', 'sub_channel']:
            assert isinstance(dim, int) or dim == "all"
            self.dim = None
            if dim != "all":
                self.dim = dim
        elif reduction not in ['section', 'global_label_corr', 'channel', 'sub_channel']:
            assert dim is None
            self.dim = "do not use"
        else:
            raise ValueError(f"unexpected reduction: {reduction}")
        assert reduction in ['mean', 'section', 'global_label_corr', 'channel', 'sub_channel']
        assert isinstance(channels, list)
        assert isinstance(key_prefix, str)

    def forward(
        self,
        x_dict,
        y_dict,
        loss_weight,
        # group_valid,
        centerness=None,
    ):
        a = x_dict[self.pred_key][:, :, self.channels, :].clone()
        b = y_dict[self.label_key][:, :, self.channels, :].clone()
        if self.reduction == 'mean':
            return 1 - global_corr(
                a,
                b,
                loss_weight,
                self.ra,
                # group_valid,
            )
        elif self.reduction == 'section':
            return 1 - section_corr(
                a,
                b,
                self.dim,
                self.ra,
                loss_weight,
                # group_valid,
            )
        elif self.reduction == 'global_label_corr':
            return 1 - global_label_corr(
                a,
                b,
                self.dim,
                loss_weight,
            )
        # elif self.reduction == 'channel':
        #     return 1 - glosec_corr(a, b, self.dim)
        # elif self.reduction == 'sub_channel':
        #     _, _, C, _ = a.shape
        #     for c in range(C - 1, 0, -1):
        #         a[:, :, c, :] -= a[:, :, c - 1, :]
        #         b[:, :, c, :] -= b[:, :, c - 1, :]
        #     return 1 - glosec_corr(a, b, self.dim)
        else:
            raise NotImplementedError
            _, _, seq_len = a.shape
            centerness = centerness.repeat(1, seq_len)
            centerness = centerness.flatten()
            a = a.flatten()
            b = b.flatten()
            numerator = weighted_mean(centerness, a * b) - weighted_mean(centerness, a) * weighted_mean(centerness, b)
            denominator = weighted_std(centerness, a) * weighted_std(centerness, b)
            corr = numerator / denominator
            return 1 - corr


class XYLoss(nn.Module):

    def __init__(self, channels=None, key_prefix=None, weight=None, ra=None):
        super().__init__()
        self.channels = channels
        self.key_prefix = key_prefix
        self.label_key = f"{key_prefix}label"
        self.pred_key = f"{key_prefix}pred"
        self.ra = ra
        self.weight = weight
        assert isinstance(weight, list)
        assert isinstance(channels, list)
        assert len(weight) == len(channels)
        assert isinstance(key_prefix, str)

    def forward(
        self,
        x_dict,
        y_dict,
        loss_weight,
        # group_valid,
    ):
        a = x_dict[self.pred_key][:, :, self.channels, :].clone()
        b = y_dict[self.label_key][:, :, self.channels, :].clone()
        B, N, C, T = a.shape
        valid = ~b.isnan()
        a = a.masked_fill(~valid, 0)
        b = b.masked_fill(~valid, 0)

        corr = 0
        for channel in range(C):
            a_c = a[:, :, channel, :]
            b_c = b[:, :, channel, :]
            a_c = (a_c - a_c.flatten().mean()) / a_c.flatten().std()
            b_c = (b_c - b_c.flatten().mean()) / b_c.flatten().std()
            corr += (a_c * b_c).mean(dim=1).flatten() * self.weight[channel] / sum(self.weight)
        corr = corr.mean() - self.ra * corr.std()
        return 1 - corr


class MultipleLoss(nn.Module):

    def __init__(self, loss_list):
        super().__init__()
        self.losses = nn.ModuleList()
        self.loss_weights = []
        for single_loss in loss_list:
            self.losses.append(eval(single_loss["name"])(**single_loss["params"]))
            self.loss_weights.append(single_loss["weight"])

    def forward(
        self,
        pred,
        label,
        loss_weight,
        # group_valid,
    ):
        losses = []
        for loss, weight in zip(self.losses, self.loss_weights):
            losses.append(weight * loss(
                pred,
                label,
                loss_weight,
                # group_valid,
            ))
        return sum(losses)
