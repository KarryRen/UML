# -*- coding: utf-8 -*-
# @Time    : 2023/3/21 16:29
# @Author  : Karry Ren

""" The modules for UML_Net.

All modules can be divided into 2 classes:
    - The modules for UML_Net.
        1. PairWiseFeatureMixer(). The feature mixer module.
        2. MutualFeatureDecoder(). The mutual feature decoder.
        3. UNSegDecoder(). The UN guided Segmentation decoder.
        4. UncertaintyNavigator(). The Uncertainty Navigator.

    - The assisted modules.
        1. PolarizedSelfAttention(). The PolarizedSelfAttention module for PairWiseFeatureMixer.
        2. BasicConv2d(). The basic 2d Conv layer for easy using.
        3. UpConv(). The UNet-Decoder like conv layer for decoding the skip-connection.
        4. SeparableConv2d(). The Special 2d Conv layer for ClsHead().

"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


# ************************************************************************************ #
# ***************************** THE MODULES FOR UML_NET ****************************** #
# ************************************************************************************ #
class PairWiseFeatureMixer(nn.Module):
    """ The module which mix features in [cls feature list] and [seg feature list]
        to [mutual feature list] by Pairwise Channel Map Interaction.

        Ref. https://www.sciencedirect.com/science/article/abs/pii/S1568494622006184

    """

    def __init__(self):
        """ Init function of PairWiseFeatureMixer.
            For simplification, we just fix the hyperparameters in this model.

        """

        super(PairWiseFeatureMixer, self).__init__()

        # 4 groups
        self.group = 4

        # conv layers for cls_features in cls_feature_list
        self.cf_conv1 = BasicConv2d(64, 256, kernel_size=1)
        self.cf_conv2 = BasicConv2d(256, 256, kernel_size=1)
        self.cf_conv3 = BasicConv2d(512, 256, kernel_size=1)
        self.cf_conv4 = BasicConv2d(1024, 256, kernel_size=1)

        # conv layers for seg_features in seg_feature_list
        self.sf_conv1 = BasicConv2d(64, 256, kernel_size=1)
        self.sf_conv2 = BasicConv2d(256, 256, kernel_size=1)
        self.sf_conv3 = BasicConv2d(512, 256, kernel_size=1)
        self.sf_conv4 = BasicConv2d(1024, 256, kernel_size=1)

        # attention module to strength feature
        self.PSA1 = PolarizedSelfAttention(512)
        self.PSA2 = PolarizedSelfAttention(512)
        self.PSA3 = PolarizedSelfAttention(512)
        self.PSA4 = PolarizedSelfAttention(512)

        # the output conv layers to adjust channels
        self.out_conv1 = BasicConv2d(512, 64, kernel_size=1)
        self.out_conv2 = BasicConv2d(512, 256, kernel_size=1)
        self.out_conv3 = BasicConv2d(512, 512, kernel_size=1)
        self.out_conv4 = BasicConv2d(512, 1024, kernel_size=1)

    def forward(self, cls_feature_list: List[torch.Tensor], seg_feature_list: List[torch.Tensor]):
        """ Forward function of PairWiseFeatureMixer.

        :param: cls_feature_list: feature list of classification encoder
        :param: seg_feature_list: feature list of segmentation encoder

        Because the encoders of classification and segmentation are both Res2Net. The feature_list
         of classification and segmentation have the same shape items:
            - item_0, shape=(bs, 64, h, w)
            - item_1, shape=(bs, 256, h/2, w/2)
            - item_2, shape=(bs, 521, h/4, w/4)
            - item_3, shape=(bs, 1024, h/8, w/8)

        returns:
            - mutual_feature_list: feature list after feature mixing using for mutual learning
                (have the same shape items with cls and seg feature list) !

        """

        # ---- Step 1. Read the cls and seg feature list ---- #
        [cf1, cf2, cf3, cf4] = cls_feature_list
        [sf1, sf2, sf3, sf4] = seg_feature_list

        # ---- Step 2. Conv to set channel to 256 ---- #
        # - conv classification features
        cf1 = self.cf_conv1(cf1)  # channel from 64 => 256
        cf2 = self.cf_conv2(cf2)  # channel from 256 => 256
        cf3 = self.cf_conv3(cf3)  # channel from 512 => 256
        cf4 = self.cf_conv4(cf4)  # channel from 1024 => 256
        # - conv segmentation features
        sf1 = self.sf_conv1(sf1)  # channel from 64 => 256
        sf2 = self.sf_conv2(sf2)  # channel from 256 => 256
        sf3 = self.sf_conv3(sf3)  # channel from 512 => 256
        sf4 = self.sf_conv4(sf4)  # channel from 1024 => 256

        # ---- Step 3. Merge the feature of cls and seg, shuffle to mix ---- #
        merge_feats1 = torch.cat([cf1, sf1], dim=1)
        merge_feats1 = self.channel_shuffle(merge_feats1)  # shape=(bs, 512, h, w)
        merge_feats2 = torch.cat([cf2, sf2], dim=1)
        merge_feats2 = self.channel_shuffle(merge_feats2)  # shape=(bs, 512, h/2, w/2)
        merge_feats3 = torch.cat([cf3, sf3], dim=1)
        merge_feats3 = self.channel_shuffle(merge_feats3)  # shape=(bs, 512, h/4, w/4)
        merge_feats4 = torch.cat([cf4, sf4], dim=1)
        merge_feats4 = self.channel_shuffle(merge_feats4)  # shape=(bs, 512, h/8, w/8)

        # ---- Step 4. PSA and use output conv to adjust channels ---- #
        f1 = self.PSA1(merge_feats1)
        out_f1 = self.out_conv1(f1)  # 512 => 64   out_f1: (bs, 64, h, w)
        f2 = self.PSA2(merge_feats2)
        out_f2 = self.out_conv2(f2)  # 512 => 256  out_f2: (bs, 256, h/2, w/2)
        f3 = self.PSA3(merge_feats3)
        out_f3 = self.out_conv3(f3)  # 512 => 512  out_f3: (bs, 512, h/4, w/4)
        f4 = self.PSA4(merge_feats4)
        out_f4 = self.out_conv4(f4)  # 512 => 1024 out_f4: (bs, 1024, h/8, w/8)

        # ---- Step 5. List the out features and return ---- #
        mutual_feature_list = [out_f1, out_f2, out_f3, out_f4]
        return mutual_feature_list

    def channel_shuffle(self, x: torch.Tensor):
        """ Do the channel shuffle, the core operation of feature mixer.

        :param x: the feature to shuffle.

        """

        # ---- Step 1. Get the shape of feature ---- #
        batch_size, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0, "Channel shuffle ERROR !!!"

        # ---- Step 2. Use reshape to shuffle (Could be update) ---- #
        group_channels = num_channels // self.group
        x = x.reshape(batch_size, group_channels, self.group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batch_size, num_channels, height, width)

        return x


class MutualFeatureDecoder(nn.Module):
    """ The `TBraTS-liked` feature decoder of mutual_feature_list,
        which generates the layered segmentation result and `pixel-wise` uncertainty.

    """

    def __init__(self, seg_class: int):
        """ Init function of MutualFeatureDecoder.

        :param seg_class: the target class of segmentation task.

        """

        super(MutualFeatureDecoder, self).__init__()
        self.seg_classes = seg_class

        # adjust channel conv, 4 means the bottom layer
        self.conv_adj_channel4 = BasicConv2d(1024, 512)
        self.conv_adj_channel3 = BasicConv2d(512, 256)
        self.conv_adj_channel2 = BasicConv2d(256, 128)

        # UNet-structure up conv to decoding features
        self.up_conv4 = UpConv2d(512, 256)
        self.up_conv3 = UpConv2d(256, 128)
        self.up_conv2 = UpConv2d(128, 64)

        # out conv to get segmentation result layers
        self.out_conv4 = nn.Sequential(BasicConv2d(512, 256),
                                       BasicConv2d(256, self.seg_classes))
        self.out_conv3 = nn.Sequential(BasicConv2d(256, 128),
                                       BasicConv2d(128, self.seg_classes))
        self.out_conv2 = nn.Sequential(BasicConv2d(128, 64),
                                       BasicConv2d(64, self.seg_classes))
        self.out_conv1 = nn.Sequential(BasicConv2d(64, 32),
                                       BasicConv2d(32, self.seg_classes))

    def forward(self, mut_feature_list: List[torch.Tensor]):
        """ Forward function of MutualFeatureDecoder,
              compute the layered segmentation and pixel-wise uncertainty.

        :param mut_feature_list: input mutual feature list

        returns:
            - mut_evidence_list: 4 layer mutual evidence (bs, seg_class, h/2^{i-1}, w/2^{i-1})
            - mut_alpha_list: 4 layer mutual alpha (bs, seg_class, h/2^{i-1}, w/2^{i-1})
            - mut_uncertainty_list: 4 layer mutual alpha (bs, 1, h/2^{i-1}, w/2^{i-1})
            - mut_info_list: 4 layer mutual information
                ~ item_0, shape=(bs, 64, h, w)
                ~ item_1, shape=(bs, 128, h/2, w/2)
                ~ item_2, shape=(bs, 256, h/4, w/4)
                ~ item_3, shape=(bs, 512, h/8, w/8)
        """

        # ---- Step 0. Define the empty mutual list ---- #
        mut_info_list = [None] * len(mut_feature_list)  # mutual information list, using for guiding segmentation
        mut_fseg_list = [None] * len(mut_feature_list)  # mutual feature of seg list, using for generating outputs

        # ---- Step 1. Adjust feature channel ---- #
        encoding_feature4 = self.conv_adj_channel4(mut_feature_list[3])
        encoding_feature3 = self.conv_adj_channel3(mut_feature_list[2])
        encoding_feature2 = self.conv_adj_channel2(mut_feature_list[1])
        encoding_feature1 = mut_feature_list[0]

        # ---- Step 2. Use UpConv backbone to get features ---- #
        # layer 4
        x = encoding_feature4  # (bs, 512, h/8, w/8)
        mut_info_list[3] = x  # set the mutual information
        mut_fseg_list[3] = self.out_conv4(x)  # (bs, seg_class, h/8, w/8)
        # layer 3
        x = self.up_conv4(x, encoding_feature3)  # (bs, 256, h/4, w/4)
        mut_info_list[2] = x  # set the mutual information
        mut_fseg_list[2] = self.out_conv3(x)  # (bs, seg_class, h/4, w/4)
        # layer 2
        x = self.up_conv3(x, encoding_feature2)  # (bs, 128, h/2, w/2)
        mut_info_list[1] = x  # set the mutual information
        mut_fseg_list[1] = self.out_conv2(x)  # (bs, seg_class, h/2, w/2)
        # layer 1
        x = self.up_conv2(x, encoding_feature1)  # (bs, 64, h, w)
        mut_info_list[0] = x  # set the mutual information
        mut_fseg_list[0] = self.out_conv1(x)  # (bs, 64, h, w)

        # ---- Step 3. Compute the evidence, alpha and uncertainty of mutual ---- #
        mut_evidence_list = [None] * len(mut_fseg_list)  # the empty evidence list
        mut_alpha_list = [None] * len(mut_fseg_list)  # the empty alpha list
        mut_uncertainty_list = [None] * len(mut_fseg_list)  # the empty uncertainty list
        # for loop layers to compute
        for i in range(len(mut_fseg_list)):
            # -- get mutual feature evidence -- #
            mut_evidence = self.infer(mut_fseg_list[i])  # (bs, seg_class, h/2^{i-1}, w/2^{i-1})
            # -- use dirichlet distribution to get alpha -- #
            mut_alpha = mut_evidence + 1  # dirichlet, (bs, seg_class, h/2^{i-1}, w/2^{i-1})
            # -- get belief and uncertainty -- #
            # based on the evidential theory (b_1 + b_2 + ... b_c + u = 1)
            # must keep dim keep it as (bs, seg_class, h/2^{i-1}, w/2^{i-1})
            S_m = torch.sum(mut_alpha, dim=1, keepdim=True)
            # belief (bs, seg_class, h/2^{i-1}, w/2^{i-1}) pixel-wise
            mut_belief = (mut_alpha - 1) / (S_m.expand(mut_alpha.shape))
            # uncertainty (bs, 1, h/2^{i-1}, w/2^{i-1}) pixel-wise
            mut_uncertainty = self.seg_classes / S_m
            # -- set them -- #
            mut_evidence_list[i] = mut_evidence
            mut_alpha_list[i] = mut_alpha
            mut_uncertainty_list[i] = mut_uncertainty

        # ---- Step 4. Generate eta for classification ---- #
        eta_for_cls = (encoding_feature4, mut_uncertainty_list[3])

        return mut_evidence_list, mut_alpha_list, mut_uncertainty_list, mut_info_list, eta_for_cls

    def infer(self, x: torch.Tensor):
        """ Infer function of dirichlet distribution.

        :param x: the input feature, shape=(bs, seg_class, h, w)

        returns:
            - evidence: the Positive evidence, shape=(bs, seg_class, h, w)

        """

        # ---- Non negative computation --- #
        evidence = F.softplus(x)
        return evidence


class UNSegDecoder(nn.Module):
    """ The UN guided Segmentation decoder, which generates the final segmentation result. """

    def __init__(self, seg_class: int):
        """ Init function of UNSegDecoder.

        :param seg_class: the total classes of segmentation.

        """

        super(UNSegDecoder, self).__init__()

        # adjust channel num by conv
        self.conv_adj_channel4 = BasicConv2d(1024, 512, kernel_size=1)
        self.conv_adj_channel3 = BasicConv2d(512, 256, kernel_size=1)
        self.conv_adj_channel2 = BasicConv2d(256, 128, kernel_size=1)

        # the down sample
        self.down2 = nn.Upsample(scale_factor=1 / 2, mode="bilinear", align_corners=True)
        self.down4 = nn.Upsample(scale_factor=1 / 4, mode="bilinear", align_corners=True)
        self.down8 = nn.Upsample(scale_factor=1 / 8, mode="bilinear", align_corners=True)

        # the uncertainty navigator for segmentation (core modules of UNSegDecoder)
        self.un4 = UncertaintyNavigator(in_channels=256, init_channels=128)
        self.un3 = UncertaintyNavigator(in_channels=128, init_channels=64)
        self.un2 = UncertaintyNavigator(in_channels=64, init_channels=32)

        # UNet like up conv
        self.conv_up4 = UpConv2d(512, 256)
        self.conv_up3 = UpConv2d(256, 128)
        self.conv_up2 = UpConv2d(128, 64)

        # output conv
        self.conv_out = BasicConv2d(64, seg_class, kernel_size=1)

    def forward(self, seg_feature_list: List[torch.Tensor], mut_info_list: List[torch.Tensor],
                mut_uncertainty: torch.Tensor):
        """ Forward function of UNSegDecoder.

        :param seg_feature_list: the segmentation feature list, 4 layers:
            - item_0, shape=(bs, 64, h, w)
            - item_1, shape=(bs, 256, h/2, w/2)
            - item_2, shape=(bs, 521, h/4, w/4)
            - item_3, shape=(bs, 1024, h/8, w/8)
        :param mut_info_list: the mutual information list, 4 layers:
            ~ item_0, shape=(bs, 64, h, w)
            ~ item_1, shape=(bs, 128, h/2, w/2)
            ~ item_2, shape=(bs, 256, h/4, w/4)
            ~ item_3, shape=(bs, 512, h/8, w/8)
        :param mut_uncertainty: the mutual pixel-wise uncertainty, shape=(bs, 1, h, w)

        returns:
            - final_seg: the final segmentation, shape=(bs, seg_class, h, w)

        """

        # ---- Step 1. Adjust channels of encoding feature ---- #
        encoding_feature4 = self.conv_adj_channel4(seg_feature_list[3])  # (bs, 512, h/8, w/8)
        encoding_feature3 = self.conv_adj_channel3(seg_feature_list[2])  # (bs, 256, h/4, w/4)
        encoding_feature2 = self.conv_adj_channel2(seg_feature_list[1])  # (bs, 128, h/2, w/2)
        encoding_feature1 = seg_feature_list[0]  # (bs, 64, h, w)

        # ---- Step 2. Decoding feature layer by layer ---- #
        # - Layer 4 (bottom)
        feature = encoding_feature4  # (bs, 512, h/8, w/8)
        encoding_feature3 = self.un4(encoding_feature3,
                                     self.down4(mut_info_list[0]), self.down4(mut_uncertainty),
                                     4)  # (bs, 256, h/4, w/4)
        # - Layer 3
        feature = self.conv_up4(feature, encoding_feature3)  # (bs, 256, h/4, w/4)
        encoding_feature2 = self.un3(encoding_feature2,
                                     self.down2(mut_info_list[0]), self.down2(mut_uncertainty),
                                     2)  # (bs, 128, h/2, w/2)
        # - Layer 2
        feature = self.conv_up3(feature, encoding_feature2)  # (bs, 128, h/2, w/2)
        encoding_feature1 = self.un2(encoding_feature1,
                                     mut_info_list[0], mut_uncertainty,
                                     1)  # (bs, 64, h, w)
        # - Layer 1 (top)
        feature = self.conv_up2(feature, encoding_feature1)  # (bs, 64, h, w)

        # ---- Step 3. Generate the final segmentation ---- #
        final_seg = self.conv_out(feature)  # (bs, seg_class, h, w)
        return final_seg


class UNClsDecoder(nn.Module):
    """ The UN guided Classification decoder, which generates the final classification result
        and image-wise uncertainty.

    """

    def __init__(self, cls_class: int, dropout: float = 0.1):
        """ Init function of UNClsDecoder.

        :param cls_class: the total classes of classification
        :param dropout: the dropout ratio

        """

        super(UNClsDecoder, self).__init__()
        self.classes = cls_class

        # the UN for classification
        self.un = UncertaintyNavigator(in_channels=1024, init_channels=512)
        # the reliable conv
        self.conv_reliable = DoubleConv2d(in_channels=512, out_channels=1024)
        # the cls_head to extract cls_feature
        self.cls_head = nn.Sequential(
            SeparableConv2d(1024, 1024, 3, dilation=2, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            SeparableConv2d(1024, 512, 3, dilation=2, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            SeparableConv2d(512, 512, 3, dilation=2, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            SeparableConv2d(512, 1024, 3, dilation=2, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
        )
        # the average pool
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # the linear fc layer
        self.cls_predict = nn.Sequential(nn.Linear(1024, self.classes, bias=False))

    def forward(self, cls_feature: torch.Tensor, eta_for_cls: tuple):
        """ Forward function of UNClsDecoder.

        :param cls_feature: the bottom feature of `cls_feature_list`
        :param eta_for_cls: the eta from mut_feature_decoder
            0- bottom_mutual_feature, shape=(bs, 512, h/8, w/8)
            1- bottom_mutual_uncertainty, shape=(bs, 1, h/8, w/8)

        returns:
            - cls_alpha: the final classification result, shape=(bs, cls_class)
            - cls_uncertainty: the image-level uncertainty, shape=(bs, 1)

        """

        # ---- Step 1. Extract classification feature ---- #
        bs, c, h, w = cls_feature.size()
        cls_feats = self.cls_head(cls_feature)

        # ---- Step 2. Use un to get eta ---- #
        eta = self.un(cls_feats, eta_for_cls[0], eta_for_cls[1], 2)

        # ---- Step 3. Use the eta to strength cls_feature ---- #
        # get the reliable feature by multiply
        cls_reliable_feature_raw = cls_feats * eta
        # add to strength
        cls_feats_reliable = cls_feats + cls_reliable_feature_raw
        cls_feats = cls_feats_reliable

        # ---- Step 4. Do the classification prediction ---- #
        cls_feats_reliable = self.avg_pool(cls_feats_reliable)
        cls_feats_fc = cls_feats_reliable.view(bs, -1)
        cls_out = self.cls_predict(cls_feats_fc)  # fully connected

        # ---- STEP 5. Compute cls alpha and uncertainty ---- #
        # -- get cls evidence -- #
        cls_evidence = self.infer(cls_out)
        # -- use dirichlet distribution to get cls alpha -- #
        cls_alpha = cls_evidence + 1
        # -- get belief and uncertainty -- #
        # must keep dim keep it as (bs, 1) not (bs,)
        S = torch.sum(cls_alpha, dim=1, keepdim=True)
        # belief (bs, cls_class) image-level
        cls_belief = (cls_alpha - 1) / (S.expand(cls_alpha.shape))
        # uncertainty (bs, 1) image-level
        cls_uncertainty = self.classes / S
        return cls_alpha, cls_uncertainty

    def infer(self, x: torch.Tensor):
        """ Infer function of dirichlet distribution.

        :param x: the input feature, shape=(bs, cls_class, h, w)

        returns:
            - evidence: the Positive evidence, shape=(bs, cls_class, h, w)

        """

        # ---- Non negative computation --- #
        evidence = F.softplus(x)
        return evidence


class UncertaintyNavigator(nn.Module):
    """ The Attention-U uncertainty navigator. """

    def __init__(self, in_channels: int, init_channels: int):
        """ Init function of UncertaintyNavigator. Ref. https://zhuanlan.zhihu.com/p/114471013

        :param in_channels: input channel of x and mutual info
        :param init_channels: init channel of attention layer

        """

        super(UncertaintyNavigator, self).__init__()

        # weight of x
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(init_channels)
        )
        # weight of m
        self.W_m = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(init_channels)
        )
        # psi weight
        self.psi = nn.Sequential(
            nn.Conv2d(init_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, mut_info: torch.Tensor, mut_uncertainty: torch.Tensor, repeat_times: int):
        """ Forward function of UncertaintyNavigator.

        :param x: the input feature, shape=(bs, in_channels, h, w)
        :param mut_info: mutual information feature (bs, in_channels, h, w)
        :param mut_uncertainty: mutual pixel-wise uncertainty (bs, 1, h, w)
        :param repeat_times: the time of channel repeating

        returns:
            - un_feature: shape=(bs, in_channels, h, w)

        """

        # ---- Step 1. Make uncertainty_map to certainty use e^(-u) ---- #
        mut_certainty_map = torch.exp(-mut_uncertainty)

        # ---- Step 2. Uncertainty Informed Attention by channels ---- #
        x = x * mut_certainty_map

        # ---- Step 3. Repeat Channel ---- #
        mut_info = mut_info.repeat(1, repeat_times, 1, 1)

        # ---- Step 4. Mutual Informed Attention like Attention-U ---- #
        x_init = self.W_x(x)
        m_init = self.W_m(mut_info)
        psi = self.relu(x_init + m_init)
        psi = self.psi(psi)
        un_feature = x * psi
        return un_feature


# ************************************************************************************ #
# ******************************* THE ASSISTED MODULES ******************************* #
# ************************************************************************************ #
class PolarizedSelfAttention(nn.Module):
    """ PSAttention algorithm in channel shuffle for PairWiseFeatureMixer,
        use self attention mechanism, we use this as a `BLACK BOX`.

        Ref. https://www.sciencedirect.com/science/article/abs/pii/S1568494622006184

    """

    def __init__(self, channel: int = 512):
        """ Init function of PolarizedSelfAttention.

        :param channel: the input channel.

        """

        super(PolarizedSelfAttention, self).__init__()
        self.ch_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1))
        self.softmax = nn.Softmax(1)
        self.ch_wz = nn.Conv2d(channel // 2, channel, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(channel)
        self.sigmoid = nn.Sigmoid()
        self.sp_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.sp_wq = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor):
        """ Forward function of PolarizedSelfAttention.

        :param x: the input feature map, shape=(bs, c, h, w)

        return:
            - out: the output feature map, shape=(bs, c, h, w)

        """

        # ---- Get the shape of x ---- #
        b, c, h, w = x.size()

        # ---- Channel-only Self-Attention ---- #
        channel_wv = self.ch_wv(x)  # c => c // 2 : (bs, c//2, h, w)
        channel_wq = self.ch_wq(x)  # c => 1 : (bs, 1, h, w)
        channel_wv = channel_wv.reshape(b, c // 2, -1)  # (bs, c//2, h * w) flatten feature map to pixel by pixel
        channel_wq = channel_wq.reshape(b, -1, 1)  # (bs, h * w, 1)
        channel_wq = self.softmax(channel_wq)  # get weight of each pixel of channels
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)  # (bs, c//2, 1, 1)
        channel_weight = (self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))).
                          permute(0, 2, 1).reshape(b, c, 1, 1))  # (bs, c, 1, 1)
        channel_out = channel_weight * x  # Channel-only attention

        # ---- Spatial-only Self-Attention ---- #
        spatial_wv = self.sp_wv(channel_out)  # (bs, c//2, h, w) why channel_out rather than x
        spatial_wq = self.sp_wq(channel_out)  # (bs, c//2, h, w)
        spatial_wq = self.agp(spatial_wq)  # (bs, c//2, 1, 1)
        spatial_wv = spatial_wv.reshape(b, c // 2, -1)  # (bs, c//2, h*w)
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)  # (bs, 1, c//2)
        spatial_wz = torch.matmul(spatial_wq, spatial_wv)  # (bs, 1, h*w)
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))  # (bs, 1, h, w)
        spatial_out = spatial_weight * channel_out  # Spatial-only attention

        # ---- Add attention ---- #
        out = spatial_out + channel_out
        return out


class BasicConv2d(nn.Module):
    """ The Basic 2d Conv layer for easy using (set a lot of default hyperparams).
        Adjust the channels using 1x1 Conv, while extracting feature.

    You might have some questions for this 2d Conv layer:
        - Why the BasicConv2d is 1x1 Conv ?
            This is indeed a question worth exploring.
            For the time being, we have chosen 1x1 because it works better in our tests.

        - Why the BasicConv2d has no BN, Relu or Dropout ?
            This is indeed a question worth exploring.
            Now, we designed like this, just because it works better in our tests.

    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 1, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        """ Init function of BasicConv2d, set a lot of default hyperparams,

        :param in_channels: input channels num
        :param out_channels: output channels num
        :param kernel_size: the kernel size
        :param stride: the conv stride
        :param padding: whether padding
        :param dilation: the dilation ratio
        :param bias: need bias or not

        """

        super(BasicConv2d, self).__init__()

        # ---- Define the nn.Conv2d layer ---- #
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=1, bias=bias)

    def forward(self, x: torch.Tensor):
        """ Forward function of BasicConv2d, just do the convolutional operation.

        :param x: the input feature map, shape=(bs, in_channels, h, w)

        return:
            - x: the output feature map, shape=(bs, out_channels, h, w)

        """

        x = self.conv(x)
        return x


class UpConv2d(nn.Module):
    """ The UNet-Decoder like conv layer for decoding the skip-connection feature.

    You might have some questions for this 2d UpConv layer:
        - Why the 2d UpConv is different from UNet Decoder ?
            Formally, UpConv does differ from UNet, but in the context of the whole network,
              the core idea is the same. The reason for this is that it is easy to understand,
              just as the description in UNet framework.

    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        """ Init function of UpConv2d.

        :param in_channels: the input channels num
        :param out_channels: the output channels num, the out_channels must be half of in_channels
        :param bilinear: use bilinear to UpSample or not

        """

        super(UpConv2d, self).__init__()

        # UpSample type
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        # Two conv layers
        self.conv1 = DoubleConv2d(in_channels, out_channels)
        self.conv2 = DoubleConv2d(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """ Forward function of UpConv2d. Do the feature extraction.

        :param x1: the feature from down, shape=(bs, in_channels, h/n, w/n)
        :param x2: the feature of skip-connection, shape=(bs, in_channels/2, (h/n)*2, (w/n)*2)

        return:
            - output: the feature after UpConv,  shape=(bs, out_channels, (h/n)*2, (w/n)*2)

        """

        # ---- Step 1. UpSample the down feature ---- #
        x1 = self.up(x1)  # shape from (bs, in_channels, h/n, w/n) to (bs, in_channels, (h/n)*2, (w/n)*2)

        # ---- Step 2. Conv the down feature ---- #
        x1 = self.conv1(x1)  # shape from (bs, in_channels, (h/n)*2, (w/n)*2) to (bs, out_channels, (h/n)*2, (w/n)*2)

        # ---- Step3. Cat the skip-connection feature ---- #
        # shape from (bs, out_channels, (h/n)*2, (w/n)*2) to  (bs, 2*out_channels(in_channels), (h/n)*2, (w/n)*2)
        x = torch.cat([x1, x2], dim=1)

        # ---- Step 4. Conv the connection feature ---- #
        # shape from (bs, 2*out_channels(in_channels), (h/n)*2, (w/n)*2) to (bs, out_channels, (h/n)*2, (w/n)*2)
        output = self.conv2(x)
        return output


class DoubleConv2d(nn.Module):
    """ The Double 2d Conv layer for easy using. """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        """ Init function of DoubleConv2d.

        :param in_channels: the input channels num
        :param out_channels: the output channel num
        :param dropout: the dropout ration

        """

        super(DoubleConv2d, self).__init__()

        # define the double_conv layer
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout)
        )

    def forward(self, x: torch.Tensor):
        """ Forward function of DoubleConv2d. Do the feature extraction.

        :param x: the conv feature, shape=(bs, in_channels, h, w)

        returns:
            - output: the out feature, shape=(bs, out_channels, h, w)
        """

        # ---- Do the double conf ---- #
        # => Conv => BN => Relu
        # => Conv => BN => Relu
        # => dropout
        output = self.double_conv(x)
        return output


class SeparableConv2d(nn.Module):
    """ A special 2d Conv layer.
        Ref. DSI https://ieeexplore.ieee.org/document/9440441

    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1, padding: int = 0,
                 dilation: int = 1, bias: bool = False):
        """ Init function of SeparableConv2d.

        :param in_channels: input channels num
        :param out_channels: output channels num
        :param kernel_size: the kernel size
        :param stride: the conv stride
        :param padding: whether padding
        :param dilation: the dilation ratio
        :param bias: need bias or not

        """

        super(SeparableConv2d, self).__init__()

        # the traditional 2d Conv layer
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, in_channels,
                              bias=bias)
        # the 1x1 2d Conv layer
        self.point_wise_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1,
                                         bias=bias)

    def forward(self, x: torch.Tensor):
        """ Forward function of SeparableConv2d.

        :param x: the input feature, shape=(bs, in_channels, h, w)

        """

        # ---- Step 1. Do the traditional conv ---- #
        x = self.conv(x)

        # ---- Step 2. Do the point_wise conv ---- #
        x = self.point_wise_conv(x)
        return x
