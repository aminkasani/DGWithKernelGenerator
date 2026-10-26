import functools

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


class _routing(nn.Module):

    def __init__(self, in_channels, num_experts, dropout_rate):
        super(_routing, self).__init__()
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        x = torch.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return F.sigmoid(x)
    

class CondConv2D(_ConvNd):
    r"""Learn specialized convolutional kernels for each example.

    As described in the paper
    `CondConv: Conditionally Parameterized Convolutions for Efficient Inference`_ ,
    conditionally parameterized convolutions (CondConv), 
    which challenge the paradigm of static convolutional kernels 
    by computing convolutional kernels as a function of the input.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        num_experts (int): Number of experts per layer 
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor
          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor
    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
                         then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`

    .. _CondConv: Conditionally Parameterized Convolutions for Efficient Inference:
       https://arxiv.org/abs/1904.04971

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', num_experts=3, dropout_rate=0.2):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(CondConv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        self._routing_fn = _routing(in_channels, num_experts, dropout_rate)
        
        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *kernel_size))
        
        self.reset_parameters()

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
    def forward(self, inputs):
        b, _, _, _ = inputs.size()
        res = []
        for input in inputs:
            input = input.unsqueeze(0)
            pooled_inputs = self._avg_pooling(input)
            routing_weights = self._routing_fn(pooled_inputs)
            kernels = torch.sum(routing_weights[: ,None, None, None, None] * self.weight, 0)
            out = self._conv_forward(input, kernels)
            res.append(out)
        return torch.cat(res, dim=0)

from mobileNetV2_v4 import InvertedResidual
class AlMobileNetV2CondConv(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super(AlMobileNetV2CondConv, self).__init__()
        # Configuration for inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # First layer (Conv2D + BatchNorm + ReLU6)
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [nn.Sequential(
            # nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            CondConv2D(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )]

        # Building inverted residual blocks
        for t, c, n, s in self.cfgs:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # Building last layer (Conv2D + BatchNorm + ReLU6)
        self.head =nn.Sequential(
            nn.Conv2d(input_channel, self.last_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=True)
        )

        self.features = nn.Sequential(*self.features)
        self.head = nn.Sequential(*self.head)

        # Classification layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        h = self.head(x)
        x = h.mean([2, 3])  # Global Average Pooling
        h = torch.amax(h, dim=[2, 3])  # Global Max Pooling
        x = self.classifier(x)
        return x,h

# from . import ddf_mul_ext, ddf_mul_faster_ext, ddf_add_ext, ddf_add_faster_ext
# OP_DICT = {
#     'mul': ddf_mul_ext,
#     'mul_faster': ddf_mul_faster_ext,
#     'add': ddf_add_ext,
#     'add_faster': ddf_add_faster_ext
# }
# class DDFFunction(Function):
#     @staticmethod
#     def forward(ctx, features, channel_filter, spatial_filter,
#                 kernel_size=3, dilation=1, stride=1, kernel_combine='mul', version=''):
#         # check args
#         assert features.is_cuda, 'input feature must be a CUDA tensor.'
#         assert channel_filter.is_cuda, 'channel_filter must be a CUDA tensor.'
#         assert spatial_filter.is_cuda, 'spatial_filter must be a CUDA tensor.'

#         # TODO: fix CUDA code to support HALF operation
#         if features.dtype == torch.float16:
#             features = features.float()
#         if channel_filter.dtype == torch.float16:
#             channel_filter = channel_filter.float()
#         if spatial_filter.dtype == torch.float16:
#             spatial_filter = spatial_filter.float()

#         # check channel_filter size
#         b, c, h, w = features.size()
#         bc, cc, hc, wc = channel_filter.size()
#         assert bc == b and cc == c,\
#             "channel_filter size {} does not match feature size {}".format(
#                 channel_filter.size(), features.size())
#         assert hc == kernel_size and wc == kernel_size,\
#             "channel_filter size {} does not match kernel size {}".format(
#                 channel_filter.size(), kernel_size)

#         # check spatial_filter size
#         bs, cs, hs, ws, = spatial_filter.size()
#         assert bs == b and hs == h // stride and ws == w // stride,\
#             "spatial_filter size {} does not match feature size {} with stride {}".format(
#                 spatial_filter.size(), features.size(), stride)
#         assert cs == kernel_size ** 2,\
#             "spatial_filter size {} does not match kernel size {}".format(
#                 spatial_filter.size(), kernel_size)

#         assert (kernel_size - 1) % 2 == 0 and kernel_size >= 1 and dilation >= 1 and stride >= 1
#         assert kernel_combine in {'mul', 'add'}, \
#             'only support mul or add combination, instead of {}'.format(kernel_combine)

#         # record important info
#         ctx.kernel_size = kernel_size
#         ctx.dilation = dilation
#         ctx.stride = stride
#         ctx.op_type = kernel_combine

#         # build output tensor
#         output = features.new_zeros((b, c, h//stride, w//stride))

#         # choose a suitable CUDA implementation based on the input feature, filter size, and combination type.
#         if version == 'f':
#             op_type = kernel_combine + '_faster'
#         elif version == 'o':
#             op_type = kernel_combine
#         elif kernel_size <= 4 and h >= 14 and w >= 14 and stride == 1:
#             op_type = kernel_combine+'_faster'
#         else:
#             op_type = kernel_combine

#         OP_DICT[op_type].forward(features, channel_filter, spatial_filter,
#                                  kernel_size, dilation, stride, output)
#         if features.requires_grad or channel_filter.requires_grad or spatial_filter.requires_grad:
#             ctx.save_for_backward(features, channel_filter, spatial_filter)
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         assert grad_output.is_cuda

#         # TODO: support HALF operation
#         if grad_output.dtype == torch.float16:
#             grad_output = grad_output.float()

#         kernel_size = ctx.kernel_size
#         dilation = ctx.dilation
#         stride = ctx.stride
#         op_type = ctx.op_type

#         features, channel_filter, spatial_filter = ctx.saved_tensors
#         rgrad_output = torch.zeros_like(grad_output, requires_grad=False)
#         rgrad_input = torch.zeros_like(features, requires_grad=False)
#         rgrad_spatial_filter = torch.zeros_like(spatial_filter, requires_grad=False)
#         grad_input = torch.zeros_like(features, requires_grad=False)
#         grad_channel_filter = torch.zeros_like(channel_filter, requires_grad=False)
#         grad_spatial_filter = torch.zeros_like(spatial_filter, requires_grad=False)

#         # TODO: optimize backward CUDA code.
#         OP_DICT[op_type].backward(grad_output.contiguous(), features, channel_filter,
#                                   spatial_filter, kernel_size, dilation, stride,
#                                   rgrad_output, rgrad_input, rgrad_spatial_filter,
#                                   grad_input, grad_channel_filter, grad_spatial_filter)

#         return grad_input, grad_channel_filter, grad_spatial_filter, None, None, None, None, None


# ddf = DDFFunction.apply
# class FilterNorm(nn.Module):
#     def __init__(self, in_channels, kernel_size, filter_type,
#                  nonlinearity='linear', running_std=False, running_mean=False):
#         assert filter_type in ('spatial', 'channel')
#         assert in_channels >= 1
#         super(FilterNorm, self).__init__()
#         self.in_channels = in_channels
#         self.filter_type = filter_type
#         self.runing_std = running_std
#         self.runing_mean = running_mean
#         std = calculate_gain(nonlinearity) / kernel_size
#         if running_std:
#             self.std = nn.Parameter(
#                 torch.randn(in_channels * kernel_size ** 2) * std, requires_grad=True)
#         else:
#             self.std = std
#         if running_mean:
#             self.mean = nn.Parameter(
#                 torch.randn(in_channels * kernel_size ** 2), requires_grad=True)

#     def forward(self, x):
#         if self.filter_type == 'spatial':
#             b, _, h, w = x.size()
#             x = x.reshape(b, self.in_channels, -1, h, w)
#             x = x - x.mean(dim=2).reshape(b, self.in_channels, 1, h, w)
#             x = x / (x.std(dim=2).reshape(b, self.in_channels, 1, h, w) + 1e-10)
#             x = x.reshape(b, _, h, w)
#             if self.runing_std:
#                 x = x * self.std[None, :, None, None]
#             else:
#                 x = x * self.std
#             if self.runing_mean:
#                 x = x + self.mean[None, :, None, None]
#         elif self.filter_type == 'channel':
#             b = x.size(0)
#             c = self.in_channels
#             x = x.reshape(b, c, -1)
#             x = x - x.mean(dim=2).reshape(b, c, 1)
#             x = x / (x.std(dim=2).reshape(b, c, 1) + 1e-10)
#             x = x.reshape(b, -1)
#             if self.runing_std:
#                 x = x * self.std[None, :]
#             else:
#                 x = x * self.std
#             if self.runing_mean:
#                 x = x + self.mean[None, :]
#         else:
#             raise RuntimeError('Unsupported filter type {}'.format(self.filter_type))
#         return x


# def build_spatial_branch(in_channels, kernel_size, head=1,
#                          nonlinearity='relu', stride=1):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, head * kernel_size ** 2, 1, stride=stride),
#         FilterNorm(head, kernel_size, 'spatial', nonlinearity))


# def build_channel_branch(in_channels, kernel_size,
#                          nonlinearity='relu', se_ratio=0.2):
#     assert se_ratio > 0
#     mid_channels = int(in_channels * se_ratio)
#     return nn.Sequential(
#         nn.AdaptiveAvgPool2d((1, 1)),
#         nn.Conv2d(in_channels, mid_channels, 1),
#         nn.ReLU(True),
#         nn.Conv2d(mid_channels, in_channels * kernel_size ** 2, 1),
#         FilterNorm(in_channels, kernel_size, 'channel', nonlinearity, running_std=True))


# class DDFPack(nn.Module):
#     def __init__(self, in_channels, kernel_size=3, stride=1, dilation=1, head=1,
#                  se_ratio=0.2, nonlinearity='relu', kernel_combine='mul'):
#         super(DDFPack, self).__init__()
#         assert kernel_size > 1
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.dilation = dilation
#         self.head = head
#         self.kernel_combine = kernel_combine

#         self.spatial_branch = build_spatial_branch(
#             in_channels, kernel_size, head, nonlinearity, stride)

#         self.channel_branch = build_channel_branch(
#             in_channels, kernel_size, nonlinearity, se_ratio)

#     def forward(self, x):
#         b, c, h, w = x.shape
#         g = self.head
#         k = self.kernel_size
#         s = self.stride
#         channel_filter = self.channel_branch(x).reshape(b*g, c//g, k, k)
#         spatial_filter = self.spatial_branch(x).reshape(b*g, -1, h//s, w//s)
#         x = x.reshape(b*g, c//g, h, w)
#         out = ddf(x, channel_filter, spatial_filter,
#                   self.kernel_size, self.dilation, self.stride, self.kernel_combine)
#         return out.reshape(b, c, h//s, w//s)

class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)

class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)# 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output



class AlMobileNetV2Dynamic_conv(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super(AlMobileNetV2Dynamic_conv, self).__init__()
        # Configuration for inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # First layer (Conv2D + BatchNorm + ReLU6)
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [nn.Sequential(
            # nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            Dynamic_conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )]

        # Building inverted residual blocks
        for t, c, n, s in self.cfgs:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # Building last layer (Conv2D + BatchNorm + ReLU6)
        self.head =nn.Sequential(
            nn.Conv2d(input_channel, self.last_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=True)
        )

        self.features = nn.Sequential(*self.features)
        self.head = nn.Sequential(*self.head)

        # Classification layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        h = self.head(x)
        x = h.mean([2, 3])  # Global Average Pooling
        h = torch.amax(h, dim=[2, 3])  # Global Max Pooling
        x = self.classifier(x)
        return x,h


import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def entropy_from_logits(logits):
    """
    logits: [B, C]
    Returns: entropy of each sample, shape [B]
    """
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()

    entropy = -(probs * log_probs).sum(dim=1)
    return entropy


def get_bn_affine_names(model):
    """Collect names of BatchNorm weight and bias parameters."""
    names = []

    for module_name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            for param_name, _ in module.named_parameters(recurse=False):
                if param_name in ("weight", "bias"):
                    full_name = (
                        f"{module_name}.{param_name}"
                        if module_name
                        else param_name
                    )
                    names.append(full_name)

    return names


import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

def configure_bn_model(model):
    """
    Configure a model for TENT/EATA.

    Only BatchNorm affine parameters are trainable.
    Batch statistics are used instead of stored running statistics.
    """
    model.train()
    model.requires_grad_(False)

    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.requires_grad_(True)

            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None

    params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    if not params:
        raise ValueError(
            "No trainable BatchNorm parameters found. "
            "Classic TENT/EATA needs a compatible normalization layer."
        )

    return params




class TentAdapter:
    def __init__(
        self,
        model,
        lr=1e-5,
        steps=1,
    ):
        self.model = copy.deepcopy(model)
        self.params = configure_bn_model(self.model.modelW)
        # self.params = self.model.parameters()
        self.steps = steps

        self.optimizer = torch.optim.Adam(
            self.params,
            lr=lr,
        )

    @torch.enable_grad()
    def step(self, x):
        """
        Adapt on the unlabeled batch x.
        Returns logits computed before the final parameter update,
        following the usual online TENT evaluation convention.
        """
        # self.model.train()
        self.model.modelW.train()
        self.model.model.requires_grad_(False)
        self.model.model.eval()

        for _ in range(self.steps):
            self.optimizer.zero_grad(set_to_none=True)

            logits, _ = self.model(x)
            loss = entropy_from_logits(logits).mean()

            loss.backward()
            self.optimizer.step()

        return logits.detach()

@torch.enable_grad()
def compute_fisher(
    main_model,
    source_loader,
    device,
    max_samples=512,
):
    """
    Estimate empirical Fisher for BatchNorm affine parameters.

    source_loader must provide labeled source data:
        x, y

    Run this BEFORE configuring the model for TTA.
    """
    model = copy.deepcopy(main_model)
    model = model.to(device)

    was_training = model.training
    model.eval()

    names = get_bn_affine_names(model)

    if not names:
        raise ValueError("No BatchNorm affine parameters found.")

    named_params = dict(model.named_parameters())

    # Preserve original DG parameters as the EWC reference.
    reference_params = {
        name: named_params[name].detach().clone()
        for name in names
    }

    fisher = {
        name: torch.zeros_like(named_params[name])
        for name in names
    }

    # Enable gradients only for the parameters whose Fisher we need.
    original_requires_grad = {
        name: param.requires_grad
        for name, param in model.named_parameters()
    }

    for name, param in model.named_parameters():
        param.requires_grad_(name in names)

    n_samples = 0

    try:
        for batch in source_loader:
            x, y = batch[0], batch[1]

            x = x.to(device)
            y = y.to(device)

            for xi, yi in zip(x, y):
                model.zero_grad(set_to_none=True)

                logits,_ = model(xi.unsqueeze(0))
                loss = F.cross_entropy(
                    logits,
                    yi.unsqueeze(0),
                )
                loss.backward()

                for name in names:
                    grad = named_params[name].grad

                    if grad is not None:
                        fisher[name] += grad.detach().square()

                n_samples += 1

                if n_samples >= max_samples:
                    break

            if n_samples >= max_samples:
                break

    finally:
        for name, param in model.named_parameters():
            param.requires_grad_(original_requires_grad[name])

        model.train(was_training)

    if n_samples == 0:
        raise ValueError("source_loader did not provide any samples.")

    # Each entry contains (Fisher importance, original DG parameter).
    fishers = {
        name: (
            fisher[name] / n_samples,
            reference_params[name],
        )
        for name in names
    }

    return fishers

class EATAAdapter:
    def __init__(
        self,
        model,
        fishers=None,
        lr=1e-4,
        e_margin=None,
        d_margin=0.05,
        fisher_alpha=2000.0,
        prob_momentum=0.9,
    ):
        self.model = copy.deepcopy(model)
        self.params = configure_bn_model(self.model.modelW)

        self.optimizer = torch.optim.Adam(
            self.params,
            lr=lr,
        )

        # Each value is: (Fisher importance, original parameter).
        self.fishers = fishers

        # Reliability threshold. If None, use a class-count-based
        # starting heuristic on the first batch.
        self.e_margin = e_margin

        # Redundancy-filter threshold.
        self.d_margin = d_margin

        # Strength of the Fisher-based anti-forgetting penalty.
        self.fisher_alpha = fisher_alpha

        self.prob_momentum = prob_momentum
        self.current_model_probs = None

    @torch.enable_grad()
    def step(self, x):
        self.model.modelW.train()
        self.model.model.requires_grad_(False)
        self.model.model.eval()
        self.optimizer.zero_grad(set_to_none=True)

        logits,_ = self.model(x)
        probs = F.softmax(logits, dim=1)
        entropies = entropy_from_logits(logits)

        # Initialize the entropy threshold using the class count.
        if self.e_margin is None:
            self.e_margin = 0.5 * math.log(logits.shape[1])

        # Stage 1: keep reliable, low-entropy samples.
        selected_idx = torch.where(
            entropies < self.e_margin
        )[0]

        # Stage 2: remove samples redundant with historical predictions.
        if (
            selected_idx.numel() > 0
            and self.current_model_probs is not None
        ):
            similarity = F.cosine_similarity(
                probs[selected_idx],
                self.current_model_probs.unsqueeze(0),
                dim=1,
            )

            keep = similarity.abs() < self.d_margin
            selected_idx = selected_idx[keep]

        # No reliable and non-redundant samples: do not update.
        if selected_idx.numel() == 0:
            return logits.detach()

        selected_entropy = entropies[selected_idx]

        # Weight selected samples according to their entropy.
        weights = torch.exp(
            self.e_margin - selected_entropy.detach()
        )

        loss = (weights * selected_entropy).mean()

        # Fisher-based regularization against forgetting.
        if self.fishers is not None:
            ewc_loss = torch.zeros(
                (),
                device=logits.device,
            )

            named_params = dict(self.model.named_parameters())

            for name, (importance, reference) in self.fishers.items():
                if name not in named_params:
                    continue

                param = named_params[name]

                importance = importance.to(param.device)
                reference = reference.to(param.device)

                ewc_loss = ewc_loss + (
                    importance * (param - reference).square()
                ).sum()

            loss = loss + self.fisher_alpha * ewc_loss

        # Update the running mean of selected prediction probabilities.
        with torch.no_grad():
            batch_mean = probs[selected_idx].detach().mean(dim=0)

            if self.current_model_probs is None:
                self.current_model_probs = batch_mean
            else:
                self.current_model_probs = (
                    self.prob_momentum * self.current_model_probs
                    + (1.0 - self.prob_momentum) * batch_mean
                )

        loss.backward()
        self.optimizer.step()

        return logits.detach()
