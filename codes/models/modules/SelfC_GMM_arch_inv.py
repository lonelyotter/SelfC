import torch
import torch.nn as nn
import torch.nn.functional as F


class InvBlockExp(nn.Module):
    def __init__(self,
                 subnet_constructor,
                 channel_num,
                 channel_split_num,
                 clamp=1.):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num  # 3
        self.split_len2 = channel_num - channel_split_num  # 48

        self.clamp = clamp

        # F: input channel: 48, output channel: 3
        self.F = subnet_constructor(self.split_len2, self.split_len1)
        # G: input channel: 3, output channel: 48
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        # H: input channel: 3, output channel: 48
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        # input size : (bt, C + C*k^2, H/k, W/k)
        # x1 is the first 3 channels
        # x2 is the rest of the channels
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            # forward
            # y1 size: (bt, 3, H/k, W/k)
            y1 = x1 + self.F(x2)
            # s size: (bt, 48, H/k, W/k)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            # y2 size: (bt, 48, H/k, W/k)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            # backward
            # reverse calculation of forward
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]


class PixelUnshuffle(nn.Module):
    '''
    PixelUnshuffle is the reverse operation of PixelShuffle.
    It receive a tensor with shape (N, C, H, W) as input,
    and return a tensor with shape (N, C*r^2, H//r, W//r) as output.
    '''
    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        N, C, H, W = x.size()
        R = self.scale
        # (N, C, H//r, r, W//r, r)
        x = x.view(N, C, H // R, R, W // R, R)
        # (N, r, r, C, H//r, W//r)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        # (N, C*r^2, H//r, W//r)
        x = x.view(N, C * R * R, H // R, W // R)
        return x


class FrequencyAnalyzer(nn.Module):
    '''
    FrequencyAnalyzer is the non-trainable module of the Analyzer.
    It receive a tensor with shape (bt, C, H, W) as input,
    and return a tensor with shape (bt, C + C*k^2, H/k, W/k) as output.
    '''
    def __init__(self, channel_in):
        super(FrequencyAnalyzer, self).__init__()
        k = 4
        self.bicubic_down = nn.Upsample(scale_factor=(1 / k, 1 / k),
                                        mode="area")
        self.pixel_unshuffle = PixelUnshuffle(k)

        self.bicubic_up = nn.Upsample(scale_factor=(k, k), mode="area")
        self.pixel_shuffle = nn.PixelShuffle(k)

    def forward(self, x, rev=False):
        if not rev:  # forward analyzer
            # input x size: (bt, C, H, W)
            # low frequency component size: (bt, C, H/k, W/k)
            component_low_f = self.bicubic_down(x)
            # high frequency component size: (bt, C*k^2, H/k, W/k)
            component_high_f = self.pixel_unshuffle(
                x - self.bicubic_up(component_low_f))
            # output size: (bt, C + C*k^2, H/k, W/k)
            return torch.cat((component_low_f, component_high_f), dim=1)
        else:  # backward synthesizer
            # input x size: (bt, C + C*k^2, H/k, W/k)
            # low frequency component size: (bt, C, H/k, W/k)
            component_low_f = x[:, 0:3]
            # high frequency component size: (bt, C*k^2, H/k, W/k)
            component_high_f = x[:, 3:]
            # output size: (bt, C, H, W)
            return self.bicubic_up(component_low_f) + self.pixel_shuffle(
                component_high_f)


from .Subnet_constructor import DenseBlock, DenseBlock3D, DenseBlockVideoInput, FeatureCalapseBlock, D2DTInput
from torch.autograd.variable import Variable
import torch.distributions as D

import torchvision.ops


class GroupedGlobalDeformAgg(nn.Module):
    def __init__(self, c, T=5):
        super(GroupedGlobalDeformAgg, self).__init__()
        self.g = 4
        self.global_context_per_group = T * (c // self.g)
        self.global_context_reallocator = nn.Sequential(
            nn.Conv2d(self.global_context_per_group,
                      self.global_context_per_group,
                      3,
                      1,
                      1,
                      bias=True), nn.LeakyReLU(0.2),
            nn.Conv2d(self.global_context_per_group,
                      self.global_context_per_group,
                      3,
                      1,
                      1,
                      bias=True))
        nn.init.constant_(self.global_context_reallocator[2].weight, 0.)
        nn.init.constant_(self.global_context_reallocator[2].bias, 0.)
        in_channels = c
        kernel_size = 3
        stride = 1
        self.padding = 1
        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size * kernel_size * T,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size * kernel_size * T,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=c,
                                      out_channels=c,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=True)
        nn.init.constant_(self.regular_conv.weight, 0.)
        nn.init.constant_(self.regular_conv.bias, 0.)
        # self.proj1 = nn.Conv2d(in_channels=c,
        #                               out_channels=c,
        #                               kernel_size=1,
        #                               stride=1,
        #                               padding=0,
        #                               bias=False)
        # nn.init.constant_(self.proj1.weight, 0.)

    def forward(self, x):
        ### BT 64 w h
        TEMP_LEN = GlobalVar.get_Temporal_LEN()
        bt, c, h, w = x.size()
        t = TEMP_LEN
        b = bt // t
        g = self.g
        x_grouped = x.reshape(b, t, g, c // g, h, w)
        x_rearranged = x.reshape(b,t,g,c//g,h,w).transpose(1,2)\
            .reshape(b*g,self.global_context_per_group,h,w)
        x_global_enhanced = self.global_context_reallocator(x_rearranged)
        x_enhanced = x_rearranged + x_global_enhanced  ### global feature add to the original

        x_enhanced = x_enhanced.reshape(b,g,t,c//g,h,w).permute(0,2,1,3,4,5)\
            .reshape(b*t,c,h,w)

        x_stacked = x_enhanced
        offset = self.offset_conv(
            x_stacked)  #.clamp(-max_offset, max_offset) ## b T (T 2KK ) h w
        modulator = 2. * torch.sigmoid(
            self.modulator_conv(x_stacked))  ## b T (T KK ) h w
        offset = offset.reshape(b * (TEMP_LEN * TEMP_LEN), -1, h, w)
        modulator = modulator.reshape(b * (TEMP_LEN * TEMP_LEN), -1, h, w)
        x_repeat = x_enhanced.unsqueeze(1).repeat(
            1, TEMP_LEN, 1, 1, 1).reshape(b * (TEMP_LEN * TEMP_LEN), c, h, w)
        x_repeat = torchvision.ops.deform_conv2d(
            input=x_repeat,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            padding=self.padding,
            mask=modulator,
            stride=1,
        )
        x_repeat = x_repeat.reshape(b * (TEMP_LEN), TEMP_LEN, c, h, w)
        x_repeat = x_repeat.sum(1)
        # x_weight = (self.proj1(x_enhanced) )

        return x_enhanced + x_repeat


class DeformConvAgg(nn.Module):
    def __init__(self, c, T=5):
        super(DeformConvAgg, self).__init__()
        in_channels = c * T
        kernel_size = 3
        stride = 1
        self.padding = 1
        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size * kernel_size * T * T,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size * kernel_size * T * T,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=c,
                                      out_channels=c,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=True)

        self.proj = nn.Conv2d(in_channels=c,
                              out_channels=c,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=False)
        nn.init.constant_(self.proj.weight, 0.)

    def forward(self, x):
        ### BT 64 w h
        TEMP_LEN = GlobalVar.get_Temporal_LEN()
        bt, c, h, w = x.size()
        b = bt // TEMP_LEN
        x_stacked = x.reshape(b, TEMP_LEN * c, h, w)
        offset = self.offset_conv(
            x_stacked)  #.clamp(-max_offset, max_offset) ## b (T T 2KK ) h w
        modulator = 2. * torch.sigmoid(
            self.modulator_conv(x_stacked))  ## b (T T KK ) h w
        offset = offset.reshape(b * (TEMP_LEN * TEMP_LEN), -1, h, w)
        modulator = modulator.reshape(b * (TEMP_LEN * TEMP_LEN), -1, h, w)
        x_repeat = x.unsqueeze(1).repeat(1, TEMP_LEN, 1, 1,
                                         1).reshape(b * (TEMP_LEN * TEMP_LEN),
                                                    c, h, w)
        x_repeat = torchvision.ops.deform_conv2d(
            input=x_repeat,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            padding=self.padding,
            mask=modulator,
            stride=1,
        )
        x_repeat = x_repeat.reshape(b * (TEMP_LEN), TEMP_LEN, c, h, w)
        x_repeat = x_repeat.sum(1)
        x_repeat = self.proj(x_repeat)

        return x + x_repeat


class GlobalAgg(nn.Module):
    def __init__(self, c):
        super(GlobalAgg, self).__init__()
        self.fc = nn.Linear(32 * 32, 1)
        self.proj1 = nn.Conv2d(c, c, 1, 1, 0)
        self.proj2 = nn.Linear(c, c)
        self.proj3 = nn.Linear(c, c)

    def forward(self, x):
        # x size: (bt, 64, W, W)
        ### 64 w h

        # size: (bt, 64, H, W)
        x_proj1 = self.proj1(x)
        TEMP_LEN = GlobalVar.get_Temporal_LEN()
        B, C, H, W = x.size()
        # size: (bt, 64, 32, 32)
        x_down_sample = F.adaptive_avg_pool2d(x, output_size=(32, 32))
        # size: (bt, 64, 32 * 32)
        x_down_sample = x_down_sample.reshape(B, C, 32 * 32)
        # size: (bt, 64)
        x_down_sample = self.fc(x_down_sample).squeeze()
        # size: (b, t, 64)
        x_down_sample_video = x_down_sample.reshape(B // TEMP_LEN, TEMP_LEN, C)
        # size: (b, t, 64)
        x_down_sample_video_proj2 = self.proj2(x_down_sample_video)
        # size: (b, t, 64)
        x_down_sample_video_proj3 = self.proj3(x_down_sample_video)
        # size: (b, t, t)
        temporal_weight_matrix = torch.matmul(
            x_down_sample_video_proj2,
            x_down_sample_video_proj3.transpose(1, 2))
        #### T * T
        temporal_weight_matrix = F.softmax(temporal_weight_matrix / C, dim=-1)

        # size: (b, t, 64, H, W)
        x_proj1 = x_proj1.reshape(B // TEMP_LEN, TEMP_LEN, C, H, W)
        # size: (b, 64*H*W, t)
        x_proj1 = x_proj1.permute(0, 2, 3, 4,
                                  1).reshape(B // TEMP_LEN, C * H * W,
                                             TEMP_LEN)
        # size: (b, 64*H*W, t)
        weighted_feature = torch.matmul(x_proj1,
                                        temporal_weight_matrix)  ## b (chw) t

        # size: (bt, 64, H, W)
        return x + weighted_feature.reshape(B//TEMP_LEN,C,H,W,TEMP_LEN).\
            permute(0,4,1,2,3).reshape(B,C,H,W)


class STPNet(nn.Module):
    def __init__(self, opt):
        super(STPNet, self).__init__()

        self.global_module = opt["global_module"]  # nonlocal
        self.stp_blk_num = opt["stp_blk_num"]  # 6
        self.fh_loss = opt["fh_loss"]  # gmm
        self.scale = opt["scale"]  # 4
        self.K = opt["gmm_k"]  # 5
        self.stp_blk_num = self.stp_blk_num - 2  # 4

        c = 64
        self.local_m1 = D2DTInput(3, c, INN_init=False)
        self.local_m2 = D2DTInput(c, c, INN_init=False)

        if self.global_module == 'nonlocal':  # our method
            self.global_m1 = GlobalAgg(c)
            self.global_m2 = GlobalAgg(c)
        if self.global_module == 'deform':
            self.global_m1 = DeformConvAgg(c)
            self.global_m2 = DeformConvAgg(c)
        if self.global_module == 'grouped_global_deform':
            self.global_m1 = GroupedGlobalDeformAgg(c)
            self.global_m2 = GroupedGlobalDeformAgg(c)
        self.other_stp_modules = []
        for i in range(self.stp_blk_num):
            self.other_stp_modules += [D2DTInput(c, c, INN_init=False)]
            if self.global_module == 'nonlocal':
                self.other_stp_modules += [GlobalAgg(c)]
            if self.global_module == 'deform':
                self.other_stp_modules += [DeformConvAgg(c)]
            if self.global_module == 'grouped_global_deform':
                self.other_stp_modules += [GroupedGlobalDeformAgg(c)]

        self.other_stp_modules = nn.Sequential(*self.other_stp_modules)

        self.hf_dim = 3 * (self.scale**2)  # 48

        if self.fh_loss == "l2":
            self.tail_gmm = [
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(c, self.hf_dim, 1, 1, 0, bias=True)
            ]
            self.tail_gmm = nn.Sequential(*self.tail_gmm)
        elif self.fh_loss == "gmm":
            MLP_dim = c  # 64
            self.tail_gmm = [
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(c, MLP_dim * 2, 1, 1, 0, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(MLP_dim * 2, MLP_dim * 4, 1, 1, 0, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(MLP_dim * 4,
                          self.hf_dim * self.K * 3,
                          1,
                          1,
                          0,
                          bias=True)
            ]
            self.tail_gmm = nn.Sequential(*self.tail_gmm)
        elif self.fh_loss == "gmm_thin":
            MLP_dim = c
            self.tail_gmm = [
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(c, MLP_dim, 1, 1, 0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(MLP_dim, MLP_dim, 1, 1, 0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(MLP_dim,
                          self.hf_dim * self.K * 3,
                          1,
                          1,
                          0,
                          bias=True)
            ]
            self.tail_gmm = nn.Sequential(*self.tail_gmm)

    def forward(self, x):
        b, c, t, h, w = x.size()
        self.b = b
        self.c = c
        self.t = t
        self.h = h
        self.w = w
        b, c, t, h, w = x.size()
        # x size: (b, t, c, h, w)
        x = x.transpose(1, 2)
        # x size: (bt, c, h, w)
        x = x.reshape(b * t, c, h, w)

        # temp size: (bt, 64, h, w)
        temp = self.local_m1(x)
        if self.global_module:
            temp = self.global_m1(temp)
        temp = self.local_m2(temp)
        if self.global_module:
            temp = self.global_m2(temp)
        temp = self.other_stp_modules(temp)
        bt, c, w, h = temp.size()
        t = GlobalVar.get_Temporal_LEN()
        b = bt // t
        # size: (b, c, t, h, w)
        temp = temp.reshape(b, t, c, w, h).transpose(1, 2)
        # size: (b, 720, t, h, w)
        self.parameters = self.tail_gmm(temp)
        if self.fh_loss == "l2":
            return

        out_param = self.parameters

        b, c, t, h, w = out_param.size()
        # size: (b, 48, 5, 3, t, h, w)
        out_param = out_param.reshape(b, self.hf_dim, self.K, 3, t, h, w)
        # weight size: (b, 48, 5, t, h, w)
        pi = F.softmax(out_param[:, :, :, 0], dim=1)
        # log_scale size: (b, 48, 5, t, h, w)
        log_scale = torch.clamp(out_param[:, :, :, 1], -7, 7)
        # mean size: (b, 48, 5, t, h, w)
        mean = out_param[:, :, :, 2]

        # v size: (b, 48, 5, t, h, w)
        v = pi[:, :, :] * self.reparametrize(mean[:, :, :],
                                             log_scale[:, :, :])  # 重新参数化成正态分布
        
        # v size: (b, 48, t, h, w)
        v = v.sum(2)

        self.gmm_v = v

        # maybe useless
        out_param = self.parameters
        b, c, t, h, w = out_param.size()
        out_param = out_param.reshape(b, self.hf_dim, self.K, 3, t, h, w)
        out_param = out_param.permute(0, 1, 4, 5, 6, 2,
                                      3).reshape(-1, self.K, 3)
        pi = F.softmax(out_param[:, :, 0], dim=1)
        log_scale = torch.clamp(out_param[:, :, 2], -7, 7)
        gm_p = torch.stack((pi, out_param[:, :, 1], log_scale), dim=-1)
        weight = gm_p[:, :, 0]
        mean = gm_p[:, :, 1]
        log_var = gm_p[:, :, 2]
        # print(weight)
        # print(mean)
        # print(log_var)
        mix = D.Categorical(weight)
        comp = D.Normal(mean, torch.exp(log_var))
        self.gmm = D.MixtureSameFamily(mix, comp)

    def reparametrize(self, mu, logvar):
        # size: (b, 48, 5, t, h, w)
        std = torch.exp(logvar)
        eps = torch.cuda.FloatTensor(std.size()).fill_(0.0)
        eps.normal_()
        x = eps.mul(std).add_(mu)
        return x

    def neg_llh(self, hf):
        # this function may be useless
        b, c, t, h, w = hf.size()
        if self.fh_loss in ["gmm", 'gmm_thin']:
            hf = hf.reshape(-1)
            return -self.gmm.log_prob(hf)
        elif self.fh_loss == "l2":
            # return torch.sum((hf-self.parameters)**2)/(b*t)
            return torch.mean((hf - self.parameters)**2)

    def sample(self):
        if self.fh_loss in ["gmm", 'gmm_thin']:
            return self.gmm_v
        elif self.fh_loss == "l2":
            return self.parameters


from global_var import *


class SelfCInvNet(nn.Module):
    def __init__(self, opt, channel_in, channel_out, subnet_type, block_num,
                 down_num):
        super(SelfCInvNet, self).__init__()
        operations = []
        current_channel = channel_in
        subnet_constructor = subnet(subnet_type, "xavier")
        b = FrequencyAnalyzer(current_channel)
        operations.append(b)

        # after concat, the channel is 17 times of the input channel
        # include 3 low frequency channels, 48 high frequency channels.
        current_channel *= 17
        for i in range(down_num):  # down_num usually is 2
            for j in range(block_num[i]):
                b = InvBlockExp(subnet_constructor, current_channel,
                                channel_out)
                operations.append(b)

        # operations are the Analyzer/Synthesizer. It includes an FrequencyAnalyzer and several InvBlockExp.
        self.operations = nn.ModuleList(operations)

        self.stp_net = STPNet(opt)

    def forward(self, x, rev=False, cal_jacobian=False, lr_before_distor=None):
        out = x
        jacobian = 0

        if not rev:  # forward
            # input size: bt, c, h, w
            for op in self.operations:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            bt, c, h, w = out.size()
            t = GlobalVar.get_Temporal_LEN()
            b = bt // t
            out = out.reshape(b, t, c, h, w).transpose(1, 2)
            lf = out[:, 0:3]
            hf = out[:, 3:]
            out = out.transpose(1, 2).reshape(b * t, c, h, w)
            # self.stp_net(lf)
            # loss_c = self.stp_net.neg_llh(hf)
            loss_c = out.mean() * 0
            # out size: (bt, c+ck^2, h/k, w/k)
            return out, loss_c

        else:  # backward
            bt, c, h, w = out.size()
            t = GlobalVar.get_Temporal_LEN()

            b = bt // t
            out = out.reshape(b, t, c, h, w).transpose(1, 2)
            lr_input = x[:, 0:3]

            bt, c, h, w = lr_input.size()
            b = bt // t
            # lr_input size: (b, 3, t, h, w)
            lr_input = lr_input.reshape(b, t, c, h, w).transpose(1, 2)

            self.stp_net(lr_input)
            # recon_hf size: (b, 48, t, h, w)
            recon_hf = self.stp_net.sample()
            # out size: (b, 51, t, h, w)
            out = torch.cat((lr_input, recon_hf), dim=1)
            # out size: (bt, 51, h, w)
            out = out.transpose(1, 2).reshape(b * t, out.size(1), h, w)
            for op in reversed(self.operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            return out, recon_hf.transpose(1, 2).reshape(b * t, -1, h, w)


from models.modules.Subnet_constructor import subnet
