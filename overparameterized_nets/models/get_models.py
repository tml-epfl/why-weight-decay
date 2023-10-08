import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Normalize(nn.Module):
    def __init__(self, mu, std):
        super(Normalize, self).__init__()
        self.mu, self.std = mu, std

    def forward(self, x):
        return (x - self.mu) / self.std


class CustomReLU(nn.Module):
    def __init__(self):
        super(CustomReLU, self).__init__()
        self.collect_preact = True
        self.avg_preacts = []

    def forward(self, preact):
        if self.collect_preact:
            self.avg_preacts.append(preact.abs().mean().item())
        act = F.relu(preact)
        return act


class ModuleWithStats(nn.Module):
    def __init__(self):
        super(ModuleWithStats, self).__init__()

    def forward(self, x):
        for layer in self._model:
            if type(layer) == CustomReLU:
                layer.avg_preacts = []

        out = self._model(x)

        avg_preacts_all = [
            layer.avg_preacts for layer in self._model if type(layer) == CustomReLU
        ]
        self.avg_preact = np.mean(avg_preacts_all)
        return out


class Linear(ModuleWithStats):
    def __init__(self, n_cls, shape_in):
        n_cls = 1 if n_cls == 2 else n_cls
        super(Linear, self).__init__()
        d = int(np.prod(shape_in[1:]))
        self._model = nn.Sequential(Flatten(), nn.Linear(d, n_cls, bias=False))

    def forward(self, x):
        logits = self._model(x)
        return torch.cat([torch.zeros(logits.shape).cuda(), logits], dim=1)


class LinearTwoOutputs(ModuleWithStats):
    def __init__(self, n_cls, shape_in):
        super(LinearTwoOutputs, self).__init__()
        d = int(np.prod(shape_in[1:]))
        self._model = nn.Sequential(Flatten(), nn.Linear(d, n_cls, bias=False))


class IdentityLayer(nn.Module):
    def forward(self, inputs):
        return inputs


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        bn,
        learnable_bn,
        stride=1,
        activation="relu",
        droprate=0.0,
        gn_groups=32,
    ):
        super(PreActBlock, self).__init__()
        self.collect_preact = True
        self.activation = activation
        self.droprate = droprate
        self.avg_preacts = []
        self.bn1 = (
            nn.BatchNorm2d(in_planes, affine=learnable_bn)
            if bn
            else nn.GroupNorm(gn_groups, in_planes)
        )
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=not learnable_bn,
        )
        self.bn2 = (
            nn.BatchNorm2d(planes, affine=learnable_bn)
            if bn
            else nn.GroupNorm(gn_groups, planes)
        )
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=not learnable_bn
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                # We added this line to ensure scale invariance
                # nn.BatchNorm2d(in_planes, affine=learnable_bn),
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=not learnable_bn,
                )
            )

    def act_function(self, preact):
        if self.activation == "relu":
            act = F.relu(preact)
            # print((act == 0).float().mean().item(), (act.norm() / act.shape[0]).item(), (act.norm() / np.prod(act.shape)).item())
        else:
            assert self.activation[:8] == "softplus"
            beta = int(self.activation.split("softplus")[1])
            act = F.softplus(preact, beta=beta)
        return act

    def forward(self, x):
        out = self.act_function(self.bn1(x))
        shortcut = (
            self.shortcut(out) if hasattr(self, "shortcut") else x
        )  # Important: using out instead of x
        out = self.conv1(out)
        out = self.act_function(self.bn2(out))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        out += shortcut
        return out


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, droprate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.droprate = droprate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            (not self.equalInOut)
            and nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            or None
        )

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class BasicBlockResNet34(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockResNet34, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, droprate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, droprate
        )

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, droprate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    droprate,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, model_width=64, droprate=0.0):
        super(ResNet, self).__init__()
        self.in_planes = model_width
        self.half_prec = False
        # self.mu = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1).cuda()
        # self.std = torch.tensor((0.2471, 0.2435, 0.2616)).view(1, 3, 1, 1).cuda()
        self.mu = torch.tensor((0.0, 0.0, 0.0)).view(1, 3, 1, 1).cuda()
        self.std = torch.tensor((1.0, 1.0, 1.0)).view(1, 3, 1, 1).cuda()
        # if self.half_prec:
        #     self.mu, self.std = self.mu.half(), self.std.half()

        self.normalize = Normalize(self.mu, self.std)
        self.conv1 = nn.Conv2d(
            3, model_width, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(model_width)
        self.layer1 = self._make_layer(block, model_width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * model_width, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * model_width, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * model_width, num_blocks[3], stride=2)
        self.linear = nn.Linear(8 * model_width * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False, return_block=5):
        assert return_block in [1, 2, 3, 4, 5], "wrong return_block"
        # out = self.normalize(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        if return_features and return_block == 1:
            return out
        out = self.layer2(out)
        if return_features and return_block == 2:
            return out
        out = self.layer3(out)
        if return_features and return_block == 3:
            return out
        out = self.layer4(out)
        if return_features and return_block == 4:
            return out
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if return_features and return_block == 5:
            return out
        out = self.linear(out)
        return out


class PreActResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        n_cls,
        model_width=64,
        cuda=True,
        half_prec=False,
        activation="relu",
        droprate=0.0,
        bn_flag=True,
        learnable_bn=True,
        freeze_last_layer=False,
    ):
        super(PreActResNet, self).__init__()
        self.half_prec = half_prec
        self.bn_flag = bn_flag
        # in particular, 32 for model_width=64 as in the original GroupNorm paper
        self.gn_groups = model_width // 2
        self.learnable_bn = learnable_bn  # doesn't matter if self.bn=False
        self.freeze_last_layer = freeze_last_layer
        self.in_planes = model_width
        self.avg_preact = None
        self.activation = activation
        self.n_cls = n_cls
        # self.mu = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
        # self.std = torch.tensor((0.2471, 0.2435, 0.2616)).view(1, 3, 1, 1)
        self.mu = torch.tensor((0.0, 0.0, 0.0)).view(1, 3, 1, 1)
        self.std = torch.tensor((1.0, 1.0, 1.0)).view(1, 3, 1, 1)

        if cuda:
            self.mu, self.std = self.mu.cuda(), self.std.cuda()
        # if half_prec:
        #     self.mu, self.std = self.mu.half(), self.std.half()

        self.normalize = Normalize(self.mu, self.std)
        self.bn1 = (
            nn.BatchNorm2d(3, affine=self.learnable_bn)
            if self.bn_flag
            else nn.GroupNorm(self.gn_groups, 3)
        )
        self.conv1 = nn.Conv2d(
            3,
            model_width,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.learnable_bn,
        )
        self.layer1 = self._make_layer(block, model_width, num_blocks[0], 1, droprate)
        self.layer2 = self._make_layer(
            block, 2 * model_width, num_blocks[1], 2, droprate
        )
        self.layer3 = self._make_layer(
            block, 4 * model_width, num_blocks[2], 2, droprate
        )
        final_layer_factor = 8
        self.layer4 = self._make_layer(
            block, final_layer_factor * model_width, num_blocks[3], 2, droprate
        )
        self.bn = (
            nn.BatchNorm2d(
                final_layer_factor * model_width * block.expansion,
                affine=self.learnable_bn,
            )
            if self.bn_flag
            else nn.GroupNorm(
                self.gn_groups, final_layer_factor * model_width * block.expansion
            )
        )
        self.linear = nn.Linear(
            final_layer_factor * model_width * block.expansion,
            1 if n_cls == 2 else n_cls,
        )

        if self.freeze_last_layer:
            # We freeze the last linear layer to ensure scale invariance
            self.linear.weight.requires_grad = False
            # We freeze the last linear layer to ensure scale invariance
            self.linear.bias.requires_grad = False

    def _make_layer(self, block, planes, num_blocks, stride, droprate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    self.bn_flag,
                    self.learnable_bn,
                    stride,
                    self.activation,
                    droprate,
                    self.gn_groups,
                )
            )
            # layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False, return_block=5):
        assert return_block in [1, 2, 3, 4, 5], "wrong return_block"
        for layer in [*self.layer1, *self.layer2, *self.layer3, *self.layer4]:
            layer.avg_preacts = []

        # x = x / ((x**2).sum([1, 2, 3], keepdims=True)**0.5 + 1e-6)  # numerical stability is needed for RLAT
        out = self.normalize(x)
        # We added this following Arora's paper about exp learning rate
        out = self.bn1(out)
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out_3 = self.layer3(out)
        out_4 = self.layer4(out_3)
        out_4 = F.relu(self.bn(out_4))
        out_5 = F.avg_pool2d(out_4, 4)
        out_5 = out_5.view(out_5.size(0), -1)

        out = self.linear(out_5)
        if out.shape[1] == 1:
            out = torch.cat([torch.zeros_like(out), out], dim=1)
        if return_features:
            return out, out_3, out_4, out_5
        else:
            return out

    def moving_average(self, model, ema_coef):
        for param1, param2 in zip(self.parameters(), model.parameters()):
            param1.data *= ema_coef
            param1.data += param2.data * (1 - ema_coef)


class WideResNet(nn.Module):
    """Based on code from https://github.com/yaodongyu/TRADES"""

    def __init__(
        self, depth=28, num_classes=10, widen_factor=10, droprate=0.0, bias_last=True
    ):
        super(WideResNet, self).__init__()
        self.half_prec = False
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, droprate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, droprate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, droprate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes, bias=bias_last)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) and not m.bias is None:
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


class VGG(nn.Module):
    """
    VGG model. Source: https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/vgg.py
    (in turn modified from https://github.com/pytorch/vision.git)
    """

    def __init__(self, n_cls, half_prec, cfg, batch_norm, bias):
        super(VGG, self).__init__()
        self.half_prec = half_prec
        self.mu = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1).cuda()
        self.std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1).cuda()
        self.normalize = Normalize(self.mu, self.std)
        self.features = self.make_layers(cfg, batch_norm=batch_norm, bias=bias)
        n_out = cfg[-2]  # equal to 8*model_width
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(n_out, n_out, bias=bias),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(n_out, n_out, bias=bias),
            nn.ReLU(True),
            nn.Linear(n_out, n_cls, bias=bias),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                print(bias)
                if bias:
                    m.bias.data.zero_()

    def make_layers(self, cfg, batch_norm=False, bias=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=bias)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.normalize(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def moving_average(self, model, ema_coef):
        for param1, param2 in zip(self.parameters(), model.parameters()):
            param1.data *= ema_coef
            param1.data += param2.data * (1 - ema_coef)


def VGG16(n_cls, model_width, half_prec, batch_norm, bias):
    """VGG 16-layer model (configuration "D")"""
    w1, w2, w3, w4, w5 = (
        model_width,
        2 * model_width,
        4 * model_width,
        8 * model_width,
        8 * model_width,
    )
    # cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    cfg = [w1, w1, "M", w2, w2, "M", w3, w3, w3, "M", w4, w4, w4, "M", w5, w5, w5, "M"]
    return VGG(n_cls, half_prec, cfg, batch_norm, bias)


def TinyResNet(
    n_cls, model_width=64, cuda=True, half_prec=False, activation="relu", droprate=0.0
):
    bn_flag = True
    return PreActResNet(
        PreActBlock,
        [1, 1, 1, 1],
        n_cls=n_cls,
        model_width=model_width,
        cuda=cuda,
        half_prec=half_prec,
        activation=activation,
        droprate=droprate,
        bn_flag=bn_flag,
    )


def TinyResNetGroupNorm(
    n_cls, model_width=64, cuda=True, half_prec=False, activation="relu", droprate=0.0
):
    bn_flag = False
    return PreActResNet(
        PreActBlock,
        [1, 1, 1, 1],
        n_cls=n_cls,
        model_width=model_width,
        cuda=cuda,
        half_prec=half_prec,
        activation=activation,
        droprate=droprate,
        bn_flag=bn_flag,
    )


def ResNet18(n_cls, model_width=64):
    return ResNet(
        BasicBlockResNet34, [2, 2, 2, 2], num_classes=n_cls, model_width=model_width
    )


def PreActResNet18(
    n_cls,
    model_width=64,
    cuda=True,
    half_prec=False,
    activation="relu",
    droprate=0.0,
    bn_flag=True,
    learnable_bn=False,
    freeze_last_layer=False,
):
    return PreActResNet(
        PreActBlock,
        [2, 2, 2, 2],
        n_cls=n_cls,
        model_width=model_width,
        cuda=cuda,
        half_prec=half_prec,
        activation=activation,
        droprate=droprate,
        bn_flag=bn_flag,
        learnable_bn=learnable_bn,
        freeze_last_layer=freeze_last_layer,
    )


def PreActResNet34(
    n_cls,
    model_width=64,
    cuda=True,
    half_prec=False,
    activation="relu",
    droprate=0.0,
    bn_flag=True,
    learnable_bn=False,
    freeze_last_layer=False,
):
    return PreActResNet(
        PreActBlock,
        [3, 4, 6, 3],
        n_cls=n_cls,
        model_width=model_width,
        cuda=cuda,
        half_prec=half_prec,
        activation=activation,
        droprate=droprate,
        bn_flag=bn_flag,
        learnable_bn=learnable_bn,
        freeze_last_layer=freeze_last_layer,
    )


def PreActResNet18GroupNorm(
    n_cls, model_width=64, cuda=True, half_prec=False, activation="relu", droprate=0.0
):
    bn_flag = False  # bn_flag==False means that we use GroupNorm with 32 groups
    return PreActResNet(
        PreActBlock,
        [2, 2, 2, 2],
        n_cls=n_cls,
        model_width=model_width,
        cuda=cuda,
        half_prec=half_prec,
        activation=activation,
        droprate=droprate,
        bn_flag=bn_flag,
    )


def PreActResNet34GroupNorm(
    n_cls, model_width=64, cuda=True, half_prec=False, activation="relu", droprate=0.0
):
    bn_flag = False  # bn_flag==False means that we use GroupNorm with 32 groups
    return PreActResNet(
        PreActBlock,
        [3, 4, 6, 3],
        n_cls=n_cls,
        model_width=model_width,
        cuda=cuda,
        half_prec=half_prec,
        activation=activation,
        droprate=droprate,
        bn_flag=bn_flag,
    )


def ResNet34(n_cls, model_width=64):
    return ResNet(
        BasicBlockResNet34, [3, 4, 6, 3], num_classes=n_cls, model_width=model_width
    )


def WideResNet28(n_cls, model_width=10):
    return WideResNet(num_classes=n_cls, widen_factor=model_width)


def get_model(
    model_name,
    n_cls,
    half_prec,
    shapes_dict,
    model_width,
    activation="relu",
    droprate=0.0,
    batch_norm=False,
    freeze_last_layer=False,
    learnable_bn=False,
    bias=True,
):
    from vit_pytorch import ViT, SimpleViT

    if model_name == "vit_basic":
        model = ViT(
            image_size=32,
            patch_size=4,
            num_classes=n_cls,
            dim=model_width,
            depth=6,
            heads=16,
            mlp_dim=model_width * 2,
            dropout=0.1,
            emb_dropout=0.1,
        )
    elif model_name == "vit_exp":
        model = SimpleViT(
            image_size=32,
            patch_size=4,
            num_classes=n_cls,
            dim=model_width,
            depth=6,
            heads=16,
            mlp_dim=model_width * 2,
        )
    elif model_name == "resnet18":
        model = PreActResNet18(
            n_cls,
            model_width=model_width,
            half_prec=half_prec,
            activation=activation,
            droprate=droprate,
            bn_flag=batch_norm,
            freeze_last_layer=freeze_last_layer,
            learnable_bn=learnable_bn,
        )
    elif model_name == "resnet18_plain":
        model = ResNet18(n_cls, model_width)
    elif model_name == "resnet18_gn":
        model = PreActResNet18GroupNorm(
            n_cls,
            model_width=model_width,
            half_prec=half_prec,
            activation=activation,
            droprate=droprate,
        )
    elif model_name == "vgg16":
        assert droprate == 0.0, "dropout is not implemented for vgg16"
        model = VGG16(n_cls, model_width, half_prec, batch_norm, bias)
    elif model_name in ["resnet34", "resnet34_plain"]:
        model = ResNet34(n_cls, model_width)
    elif model_name == "resnet34_gn":
        model = PreActResNet34GroupNorm(
            n_cls,
            model_width=model_width,
            half_prec=half_prec,
            activation=activation,
            droprate=droprate,
        )
    elif model_name == "resnet34preact":
        model = PreActResNet34(
            n_cls,
            model_width=model_width,
            half_prec=half_prec,
            activation=activation,
            droprate=droprate,
            bn_flag=batch_norm,
            freeze_last_layer=freeze_last_layer,
            learnable_bn=learnable_bn,
        )
    elif model_name == "wrn28":
        model = WideResNet28(n_cls, model_width)
    elif model_name == "resnet_tiny":
        model = TinyResNet(
            n_cls,
            model_width=model_width,
            half_prec=half_prec,
            activation=activation,
            droprate=droprate,
        )
    elif model_name == "resnet_tiny_gn":
        model = TinyResNetGroupNorm(
            n_cls,
            model_width=model_width,
            half_prec=half_prec,
            activation=activation,
            droprate=droprate,
        )
    elif model_name == "linear":
        model = Linear(n_cls, shapes_dict)
    else:
        raise ValueError("wrong model")
    model.half_prec = half_prec
    return model


def init_weights(model, scale_init=0.0):
    def init_weights_linear(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            # m.weight.data.zero_()
            m.weight.data.normal_()
            m.weight.data *= scale_init / (m.weight.data**2).sum() ** 0.5
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights_he(m):
        # if isinstance(m, nn.Conv2d):
        #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #     m.weight.data.normal_(0, math.sqrt(2. / n))
        #     if m.bias is not None:
        #         m.bias.data.zero_()
        # elif isinstance(m, nn.Linear):
        #     n = m.in_features
        #     m.weight.data.normal_(0, math.sqrt(2. / n))
        #     if m.bias is not None:
        #         m.bias.data.zero_()

        # From Rice et al.
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    if model == "linear":
        return init_weights_linear
    else:
        return init_weights_he


def forward_pass_rlat(model, x, deltas, layers):
    i = 0

    def out_hook(m, inp, out_layer):
        nonlocal i
        if layers[i] == model.normalize:
            new_out = (torch.clamp(inp[0] + deltas[i], 0, 1) - model.mu) / model.std
        else:
            new_out = out_layer + deltas[i]
        i += 1
        return new_out

    handles = [layer.register_forward_hook(out_hook) for layer in layers]
    out = model(x)

    for handle in handles:
        handle.remove()
    return out


def get_rlat_layers(model, layers):
    # import ipdb;ipdb.set_trace()
    if layers == "all":
        return [
            model.normalize,
            model.conv1,
            model.layer1[0].bn1,
            model.layer1[0].conv1,
            model.layer1[0].bn2,
            model.layer1[0].conv2,
            model.layer1[1].bn1,
            model.layer1[1].conv1,
            model.layer1[1].bn2,
            model.layer1[1].conv2,
            model.layer1,
            model.layer2[0].bn1,
            model.layer2[0].conv1,
            model.layer2[0].bn2,
            model.layer2[0].conv2,
            model.layer2[1].bn1,
            model.layer2[1].conv1,
            model.layer2[1].bn2,
            model.layer2[1].conv2,
            model.layer2,
            model.layer3[0].bn1,
            model.layer3[0].conv1,
            model.layer3[0].bn2,
            model.layer3[0].conv2,
            model.layer3[1].bn1,
            model.layer3[1].conv1,
            model.layer3[1].bn2,
            model.layer3[1].conv2,
            model.layer3,
            model.layer4[0].bn1,
            model.layer4[0].conv1,
            model.layer4[0].bn2,
            model.layer4[0].conv2,
            model.layer4[1].bn1,
            model.layer4[1].conv1,
            model.layer4[1].bn2,
            model.layer4[1].conv2,
            model.layer4,
            model.bn,
        ]
    elif layers == "lpips":
        return [model.conv1, model.layer1, model.layer2, model.layer3, model.layer4]
    elif layers == "bnonly":
        return [
            model.normalize,
            model.layer1[0].bn1,
            model.layer1[0].bn2,
            model.layer1[1].bn1,
            model.layer1[1].bn2,
            model.layer2[0].bn1,
            model.layer2[0].bn2,
            model.layer2[1].bn1,
            model.layer2[1].bn2,
            model.layer3[0].bn1,
            model.layer3[0].bn2,
            model.layer3[1].bn1,
            model.layer3[1].bn2,
            model.layer4[0].bn1,
            model.layer4[0].bn2,
            model.layer4[1].bn1,
            model.layer4[1].bn2,
            model.bn,
        ]
    elif layers == "convonly":
        return [
            model.normalize,
            model.conv1,
            model.layer1[0].conv1,
            model.layer1[0].conv2,
            model.layer1[1].conv1,
            model.layer1[1].conv2,
            model.layer2[0].conv1,
            model.layer2[0].conv2,
            model.layer2[1].conv1,
            model.layer2[1].conv2,
            model.layer3[0].conv1,
            model.layer3[0].conv2,
            model.layer3[1].conv1,
            model.layer3[1].conv2,
            model.layer4[0].conv1,
            model.layer4[0].conv2,
            model.layer4[1].conv1,
            model.layer4[1].conv2,
        ]
    elif layers == "block0":
        return [model.conv1]
    elif layers == "block1":
        return [model.layer1]
    elif layers == "block2":
        return [model.layer2]
    elif layers == "block3":
        return [model.layer3]
    elif layers == "block4":
        return [model.layer4]
    else:
        raise ValueError("wrong RLAT layers")


def save_model(epoch, model, optim, log_dir, name):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
        },
        os.path.join(log_dir, "models", f"{name}_{epoch}.pth"),
    )


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
    BatchNorm buffers update (if any).
    Performs 1 epochs to estimate buffers average using train dataset.

    :param loader: train dataset loader for buffers average estimation.
    :param model: model being update
    :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        # input = input.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))
