import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# ========================================================================================
# Adaptive Denoising Block (replaces static ResBlock at the end)
# ========================================================================================
class AdaptiveDenoisingBlock(nn.Module):
    """
    Learns a residual correction and applies it adaptively via a learned gate map.
    """
    def __init__(self, channels=1):
        super(AdaptiveDenoisingBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=channels, kernel_size=3, padding=1)
        )
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = self.conv_block(x)
        gate = self.gate_conv(x)
        return x + residual * gate


NUM_BANDS = 1
SCALE_FACTOR = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========================================================================================
# Basic Conv / Deconv Blocks
# ========================================================================================
class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(input_size, output_size, kernel_size, stride, 0, bias=bias)
        )
        self.act = torch.nn.LeakyReLU()

    def forward(self, x):
        return self.act(self.conv(x))


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        self.act = torch.nn.LeakyReLU()

    def forward(self, x):
        return self.act(self.deconv(x))


# ========================================================================================
# Residual Block
# ========================================================================================
class ResBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(ResBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.residual(x)


# ========================================================================================
# Feature Extractor
# ========================================================================================
class FeatureExtract(torch.nn.Module):
    def __init__(self, in_channels=NUM_BANDS):
        super(FeatureExtract, self).__init__()
        channels = (16, 32, 64, 128, 256)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 7, 1, 3),
            ResBlock(channels[0]),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], 3, 2, 1),
            ResBlock(channels[1]),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], 3, 2, 1),
            ResBlock(channels[2]),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], 3, 2, 1),
            ResBlock(channels[3]),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(channels[3], channels[4], 3, 2, 1),
            ResBlock(channels[4]),
        )

    def forward(self, x):
        l1 = self.conv1(x)
        l2 = self.conv2(l1)
        l3 = self.conv3(l2)
        l4 = self.conv4(l3)
        l5 = self.conv5(l4)
        return [l1, l2, l3, l4, l5]


# ========================================================================================
# Significance Extraction (Attention Fusion)
# ========================================================================================
class SignificanceExtraction(nn.Module):
    def __init__(self, in_channels, ifattention=True, iftwoinput=False, outputM=False):
        super(SignificanceExtraction, self).__init__()
        self.attention = ifattention
        self.twoinput = iftwoinput
        self.outputM = outputM

        if self.attention:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(in_channels)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(in_channels)
            )
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )

    def forward(self, inputs):
        if self.attention:
            LS1, LS2, SpecFeature = inputs
            temporal_diff = self.conv1(LS2) - self.conv2(LS1)
            M1 = self.conv(temporal_diff)
            result = LS2 * M1 + SpecFeature * (1 - M1)
        else:
            _, LS2, SpecFeature = inputs
            result = 0.5 * LS2 + 0.5 * SpecFeature

        return (result, M1) if self.outputM and self.attention else result


# ========================================================================================
# Utility Functions
# ========================================================================================
def calc_mean_std(feat, eps=1e-5):
    N, C = feat.size()[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


# ========================================================================================
# Similarity Feature Refiner
# ========================================================================================
class SimilarityFeatureRefiner(nn.Module):
    def __init__(self, method='cosine'):
        super(SimilarityFeatureRefiner, self).__init__()
        assert method in ['cosine', 'corr']
        self.method = method

    def forward(self, HS_feat, HS_indices_feat, SS_feat):
        if self.method == 'cosine':
            norm_HS = F.normalize(HS_indices_feat, p=2, dim=1)
            norm_SS = F.normalize(SS_feat, p=2, dim=1)
            similarity = (norm_HS * norm_SS).sum(dim=1, keepdim=True)
        else:
            HS_mean = HS_indices_feat.mean(dim=1, keepdim=True)
            SS_mean = SS_feat.mean(dim=1, keepdim=True)
            HS_centered = HS_indices_feat - HS_mean
            SS_centered = SS_feat - SS_mean
            numerator = (HS_centered * SS_centered).sum(dim=1, keepdim=True)
            denominator = torch.sqrt((HS_centered ** 2).sum(dim=1, keepdim=True) *
                                     (SS_centered ** 2).sum(dim=1, keepdim=True) + 1e-6)
            similarity = numerator / denominator

        refined = HS_feat * similarity
        return refined


# ========================================================================================
# Combined Feature Generator (Main Model)
# ========================================================================================
class CombinFeatureGenerator(nn.Module):
    def __init__(self, NUM_BANDS=NUM_BANDS, ifAdaIN=True, ifAttention=True, ifTwoInput=False, outputM=False):
        super(CombinFeatureGenerator, self).__init__()

        self.ifAdaIN = ifAdaIN
        self.ifAttention = ifAttention
        self.ifTwoInput = ifTwoInput
        self.outputM = outputM

        self.indices_SNet = FeatureExtract(in_channels=3)
        self.MODIS_SNet = FeatureExtract(in_channels=1)
        self.Landsat_SNet = FeatureExtract(in_channels=1)
        self.similarity_refiner = SimilarityFeatureRefiner(method='cosine')

        channels = (16, 32, 64, 128, 256)
        self.SignE_List = nn.ModuleList(
            [SignificanceExtraction(ch, ifattention=ifAttention, outputM=outputM) for ch in channels]
        )

        self.conv1 = nn.Sequential(
            DeconvBlock(channels[4] * 2, channels[3], 4, 2, 1, bias=True),
            ResBlock(channels[3]),
        )
        self.conv2 = nn.Sequential(
            DeconvBlock(channels[3] * 2, channels[2], 4, 2, 1, bias=True),
            ResBlock(channels[2]),
        )
        self.conv3 = nn.Sequential(
            DeconvBlock(channels[2] * 2, channels[1], 4, 2, 1, bias=True),
            ResBlock(channels[1]),
        )
        self.conv4 = nn.Sequential(
            DeconvBlock(channels[1] * 2, channels[0], 4, 2, 1, bias=True),
            ResBlock(channels[0]),
        )
        self.conv5 = nn.Sequential(
            ResBlock(channels[0] * 2),
            nn.Conv2d(channels[0] * 2, channels[0], 1, 1, 0),
            ResBlock(channels[0]),
            nn.Conv2d(channels[0], NUM_BANDS, 1, 1, 0),
        )

        self.denoiser = AdaptiveDenoisingBlock(channels=NUM_BANDS)

    def forward(self, inputs):
        modis_t2, landsat_t1, sentinel_t1, modis_t1 = inputs
        target_H, target_W = sentinel_t1.shape[-2:]

        landsat_LST_t1 = landsat_t1[:, 0:1, :, :]
        landsat_indices_t1 = landsat_t1[:, 1:, :, :]

        modis_t1_10m = F.interpolate(modis_t1, size=(target_H, target_W), mode='bicubic', align_corners=False)
        modis_t2_10m = F.interpolate(modis_t2, size=(target_H, target_W), mode='bicubic', align_corners=False)
        landsat_LST_t1_10m = F.interpolate(landsat_LST_t1, size=(target_H, target_W), mode='bicubic', align_corners=False)
        landsat_indices_t1_10m = F.interpolate(landsat_indices_t1, size=(target_H, target_W), mode='bicubic', align_corners=False)

        LS1_List = self.MODIS_SNet(modis_t1_10m)
        LS2_List = self.MODIS_SNet(modis_t2_10m)
        HS_List = self.Landsat_SNet(landsat_LST_t1_10m)
        HS_indices_LIST = self.indices_SNet(landsat_indices_t1_10m)
        SS1_List = self.indices_SNet(sentinel_t1)

        new_10mHS_list = []
        for hs_lst, hs_indices, ss_indices in zip(HS_List, HS_indices_LIST, SS1_List):
            refined_hs1 = self.similarity_refiner(hs_lst, hs_indices, ss_indices)
            new_10mHS_list.append(refined_hs1)

        SpecFeature_List = []
        for refined_hs1, ls1 in zip(new_10mHS_list, LS1_List):
            if self.ifAdaIN:
                SpecFeature_List.append(adaptive_instance_normalization(refined_hs1, ls1))
            else:
                SpecFeature_List.append(refined_hs1)

        FusionFeature_List = []
        M = []
        for SignE, spec_feature, ls1, ls2 in zip(self.SignE_List, SpecFeature_List, LS1_List, LS2_List):
            result = SignE([ls1, ls2, spec_feature])
            if isinstance(result, tuple):
                FusionFeature_List.append(result[0])
                M.append(result[1])
            else:
                FusionFeature_List.append(result)

        l5_in = torch.cat((FusionFeature_List[4], LS1_List[4]), dim=1)
        l5 = self.conv1(l5_in)
        l4_in = torch.cat((FusionFeature_List[3], l5), dim=1)
        l4 = self.conv2(l4_in)
        l3_in = torch.cat((FusionFeature_List[2], l4), dim=1)
        l3 = self.conv3(l3_in)
        l2_in = torch.cat((FusionFeature_List[1], l3), dim=1)
        l2 = self.conv4(l2_in)
        l1_in = torch.cat((FusionFeature_List[0], l2), dim=1)
        l1 = self.conv5(l1_in)

        out = self.denoiser(l1)
        return (out, M) if self.outputM else out


# ========================================================================================
# GAN Loss + Discriminator
# ========================================================================================
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.Tensor = tensor
        self.loss = nn.MSELoss() if use_lsgan else nn.BCELoss()

    def get_target_tensor(self, input_tensor, target_is_real):
        if target_is_real:
            return torch.ones_like(input_tensor) * self.real_label
        else:
            return torch.zeros_like(input_tensor) * self.fake_label

    def __call__(self, input, target_is_real):
        prediction_map = input[-1] if isinstance(input, list) else input
        target_tensor = self.get_target_tensor(prediction_map, target_is_real)
        return self.loss(prediction_map, target_tensor)


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)
