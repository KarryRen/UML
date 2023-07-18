import torch
import torch.nn as nn
from models.lib.convs import SeparableConv2d, BasicConv2d, UNet_Up, DoubleConv
import torch.nn.functional as F


class PolarizedSelfAttention(nn.Module):
    """
        PSA algorithm in channel shuffle, use self attention mechanism
    """

    def __init__(self, channel=512):
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

    def forward(self, x):
        """
        Args:
            x: features

        Returns:
            features after attention
        """
        b, c, h, w = x.size()

        # ---- Channel-only Self-Attention ---- #
        channel_wv = self.ch_wv(x)  # c => c // 2 : (bs, c//2, h, w)
        channel_wq = self.ch_wq(x)  # c => 1 : (bs, 1, h, w)
        channel_wv = channel_wv.reshape(b, c // 2, -1)  # (bs, c//2, h * w) flatten feature map to pixel by pixel
        channel_wq = channel_wq.reshape(b, -1, 1)  # (bs, h * w, 1)
        channel_wq = self.softmax(channel_wq)  # get weight of each pixel of channels
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)  # (bs, c//2, 1, 1)
        channel_weight = self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))) \
            .permute(0, 2, 1).reshape(b, c, 1, 1)  # (bs, c, 1, 1)
        channel_out = channel_weight * x

        # ---- Spatial-only Self-Attention ---- #
        spatial_wv = self.sp_wv(channel_out)  # (bs, c//2, h, w) why channel_out rather than x ?
        spatial_wq = self.sp_wq(channel_out)  # (bs, c//2, h, w)
        spatial_wq = self.agp(spatial_wq)  # (bs, c//2, 1, 1)
        spatial_wv = spatial_wv.reshape(b, c // 2, -1)  # (bs, c//2, h*w)
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)  # (bs, 1, c//2)
        spatial_wz = torch.matmul(spatial_wq, spatial_wv)  # (bs, 1, h*w)
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))  # (bs, 1, h, w)
        spatial_out = spatial_weight * channel_out

        out = spatial_out + channel_out

        return out


class PairWiseFeatureMixer(nn.Module):
    """
        Mix features of (cls feature list) and (seg feature list)
        To (mutual feature list)
        By Pairwise Channel Map Interaction
    """

    def __init__(self):
        super(PairWiseFeatureMixer, self).__init__()
        self.group = 4

        # conv layers for cls_features
        self.cf_conv1 = BasicConv2d(64, 256, kernel_size=1)
        self.cf_conv2 = BasicConv2d(256, 256, kernel_size=1)
        self.cf_conv3 = BasicConv2d(512, 256, kernel_size=1)
        self.cf_conv4 = BasicConv2d(1024, 256, kernel_size=1)

        # conv layers for seg_features
        self.sf_conv1 = BasicConv2d(64, 256, kernel_size=1)
        self.sf_conv2 = BasicConv2d(128, 256, kernel_size=1)
        self.sf_conv3 = BasicConv2d(256, 256, kernel_size=1)
        self.sf_conv4 = BasicConv2d(512, 256, kernel_size=1)

        # attention Module to strength feature
        self.PSA1 = PolarizedSelfAttention(512)
        self.PSA2 = PolarizedSelfAttention(512)
        self.PSA3 = PolarizedSelfAttention(512)
        self.PSA4 = PolarizedSelfAttention(512)

        # the final convs to adjust channels
        self.f_conv1 = BasicConv2d(512, 64, kernel_size=1)
        self.f_conv2 = BasicConv2d(512, 128, kernel_size=1)
        self.f_conv3 = BasicConv2d(512, 256, kernel_size=1)

    def forward(self, cls_feature_list, seg_feature_list):
        """
        Args:
            cls_feature_list: feature list of classification encoding
            seg_feature_list: feature list of segmentation encoding

        Returns:
            feature list after feature mixing
        """
        [cf1, cf2, cf3, cf4] = cls_feature_list
        [sf1, sf2, sf3, sf4] = seg_feature_list

        # ---- set channel to 256 ---- #
        # '=>' pres channel changes no shape change
        cf1 = self.cf_conv1(cf1)  # 64 => 256
        cf2 = self.cf_conv2(cf2)  # 256 => 256
        cf3 = self.cf_conv3(cf3)  # 512 => 256
        cf4 = self.cf_conv4(cf4)  # 1024 => 256

        sf1 = self.sf_conv1(sf1)  # 64 => 256
        sf2 = self.sf_conv2(sf2)  # 128 => 256
        sf3 = self.sf_conv3(sf3)  # 256 => 256
        sf4 = self.sf_conv4(sf4)  # 512 => 256

        # ---- merge and shuffle ---- #
        merge_feats1 = torch.cat([cf1, sf1], dim=1)
        merge_feats1 = self.channel_shuffle(merge_feats1)
        merge_feats2 = torch.cat([cf2, sf2], dim=1)
        merge_feats2 = self.channel_shuffle(merge_feats2)
        merge_feats3 = torch.cat([cf3, sf3], dim=1)
        merge_feats3 = self.channel_shuffle(merge_feats3)
        merge_feats4 = torch.cat([cf4, sf4], dim=1)
        merge_feats4 = self.channel_shuffle(merge_feats4)

        # ---- split | actually no need to do split ---- #
        f1 = merge_feats1
        f2 = merge_feats2
        f3 = merge_feats3
        f4 = merge_feats4

        # ---- PSA and use Final conv to set channels ---- #
        f1 = self.PSA1(f1)
        out_f1 = self.f_conv1(f1)  # 512 => 64 out_f1: (bs, 64, h, w)
        f2 = self.PSA2(f2)
        out_f2 = self.f_conv2(f2)  # 512 => 128 out_f2: (bs, 128, h/2, w/2)
        f3 = self.PSA3(f3)
        out_f3 = self.f_conv3(f3)  # out_f3: (bs, 256, h/4, w/4)
        f4 = self.PSA4(f4)  # out_f4: (bs, 512, h/8, w/8)
        out_f4 = f4

        mixed_feature_list = [out_f1, out_f2, out_f3, out_f4]

        return mixed_feature_list

    def channel_shuffle(self, x):
        batch_size, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group

        x = x.reshape(batch_size, group_channels, self.group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batch_size, num_channels, height, width)

        return x


class TrustedMutualFeatureDecoder(nn.Module):
    """
        Trusted decoding joint feature to get the
            - pixel-wise uncertainty map (bs, 1, h, w)
            - trusted seg result (bs, c, h, w)
    """

    def __init__(self, seg_classes):
        super(TrustedMutualFeatureDecoder, self).__init__()
        self.classes = seg_classes

        # Backbone Unet decoder structure to up_sample features
        self.up_conv1 = UNet_Up(512, 256)
        self.up_conv2 = UNet_Up(512, 128)
        self.up_conv3 = UNet_Up(256, 64)
        self.out_conv1 = BasicConv2d(128, 32, kernel_size=1)
        self.out_conv2 = BasicConv2d(32, self.classes, kernel_size=1)

    def forward(self, mutual_feature_list):
        """
        Args:
            mutual_feature_list: feature list after feature mixing

        Returns:
            mutual_evidence: the pixel-wise mutual evidence (bs, c, h, w)
            mutual_alpha: the pixel-wise Alpha of mutual Dirichlet  (bs, c, h, w)
            mutual_uncertainty: the pixel-wise uncertainty of segmentation (bs, 1, h, w)
        """

        # ---- STEP 1. use Backbone to get final feature ---- #
        x = mutual_feature_list[3]  # bottom layer feature (bs, 512, h/8, w/8)
        x = self.up_conv1(x, mutual_feature_list[2])  # (bs, 512, h/4, w/4)
        x = self.up_conv2(x, mutual_feature_list[1])  # (bs, 256, h/2, w/2)
        x = self.up_conv3(x, mutual_feature_list[0])  # (bs, 128, h, w)
        x = self.out_conv1(x)  # (bs, 32, h, w)
        mutual_feature_out = self.out_conv2(x)  # (bs, c, h, w)

        # ---- STEP 2. get seg evidence ---- #
        mutual_evidence = self.infer(mutual_feature_out)  # (bs, c, h, w)

        # ---- STEP 3. use dirichlet distribution to get alpha ---- #
        mutual_alpha = mutual_evidence + 1  # dirichlet (bs, c, h, w)

        # ---- STEP 4. get belief and uncertainty ---- #
        # based on the evidential theory (b_1 + b_2 + ... b_c + u = 1)
        S = torch.sum(mutual_alpha, dim=1, keepdim=True)  # must keep dim keep it as (bs, 1, h, w) not (bs, h, w)
        seg_belief = (mutual_alpha - 1) / (S.expand(mutual_alpha.shape))  # belief (bs, c, h, w) pixel-wise
        mutual_uncertainty = self.classes / S  # uncertainty (bs, 1, h, w) pixel-wise

        return mutual_evidence, mutual_alpha, mutual_uncertainty

    def infer(self, input):
        evidence = F.softplus(input)
        return evidence


class UN_ModulePlus(nn.Module):
    """
        use mutual evidence and uncertainty to guide segmentation
        | by original_feature + reliable_mask |
    """

    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d, dropout=0.1):
        super(UN_ModulePlus, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout))
        self.conv_out = BasicConv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x, seg, mutual_evidence, mutual_uncertainty):
        """
        Args:
            x: input feature
            seg: input segmentation (bs, c, h/n, w/n)
            mutual_evidence: input mutual evidence (bs, c, h/n, w/n)
            mutual_uncertainty: input pixel-wise mutual uncertainty (bs, 1, h/n, w/n)

        Returns:
            feature_out -> for top using
            reliable_mask -> for viewing
        """

        # ---- make uncertainty_map to certainty map use (e(-u)) ---- #
        certainty_mutual_feature = torch.exp(-mutual_uncertainty)

        # ---- generate reliable mask ---- #
        reliable_mask = (seg + mutual_evidence) * certainty_mutual_feature  # by channels
        reliable_feature = self.conv(reliable_mask)

        # ---- generate out feature ---- #
        feature_out = self.conv_out(x + reliable_feature)

        return feature_out, reliable_mask


class UN_ModuleConcat(nn.Module):
    """
        use mutual evidence and uncertainty to guide segmentation
        | by Concat(original_feature, reliable_mask) |
    """

    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d, dropout=0.1):
        super(UN_ModuleConcat, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout))
        self.conv_out = BasicConv2d(2 * out_channels, out_channels, kernel_size=1)

    def forward(self, x, seg, mutual_evidence, mutual_uncertainty):
        """
         Args:
             x: input feature
             seg: input segmentation (bs, c, h/n, w/n)
             mutual_evidence: input mutual evidence (bs, c, h/n, w/n)
             mutual_uncertainty: input pixel-wise mutual uncertainty (bs, 1, h/n, w/n)

         Returns:
             feature_out -> for top using
             reliable_mask -> for viewing
         """

        # ---- make uncertainty_map to certainty map use (e(-u)) ---- #
        certainty_mutual_feature = torch.exp(-mutual_uncertainty)

        # ---- generate reliable mask ---- #
        reliable_mask = (seg + mutual_evidence) * certainty_mutual_feature  # by channels
        reliable_feature = self.conv(reliable_mask)

        # ---- generate out feature ---- #
        feature_out = self.conv_out(torch.cat((x, reliable_feature), dim=1))

        return feature_out, reliable_mask


class UncertaintyNavigatorForSeg(nn.Module):
    """
        our UN
    """

    def __init__(self, seg_classes=2):
        super(UncertaintyNavigatorForSeg, self).__init__()

        self.down2 = nn.Upsample(scale_factor=1 / 2, mode='bilinear', align_corners=True)
        self.down4 = nn.Upsample(scale_factor=1 / 4, mode='bilinear', align_corners=True)
        self.down8 = nn.Upsample(scale_factor=1 / 8, mode='bilinear', align_corners=True)

        self.conv_bottom = BasicConv2d(1024, 512, kernel_size=1)

        # cat attention
        self.UN_C3 = UN_ModuleConcat(in_channels=seg_classes, out_channels=512)
        self.UN_C0 = UN_ModuleConcat(in_channels=seg_classes, out_channels=128)

        self.conv_up3 = UNet_Up(512, 256)
        self.conv_up2 = UNet_Up(512, 128)
        self.conv_up1 = UNet_Up(256, 64)

        self.conv_out3 = BasicConv2d(512, seg_classes, kernel_size=1)
        self.conv_out2 = BasicConv2d(512, seg_classes, kernel_size=1)
        self.conv_out1 = BasicConv2d(256, seg_classes, kernel_size=1)
        self.conv_out0 = BasicConv2d(128, seg_classes, kernel_size=1)

    def forward(self, seg_feature_list, bottom_mut_feature, mutual_evidence, mutual_uncertainty):
        """
        Args:
            seg_feature_list: feature list of segmentation
            bottom_mut_feature: bottom feature of mutual feature
            mutual_evidence: mutual seg evidence
            mutual_uncertainty: mutual pixel-wise uncertainty

        Returns:
            seg_list: hyrical segmentation result (4 layers)
            reliable_mask_list: reliable mask (just for viewing)
            eta: the reliable feature from segmentation to classification
        """

        seg_list = [None] * len(seg_feature_list)
        reliable_mask_list = [None] * len(seg_feature_list)

        # ---- initial feature process ---- #
        bottom_seg_feature = seg_feature_list[3]  # last layer (bs, 512, h/8, w/8)
        feature = torch.cat([bottom_seg_feature, bottom_mut_feature], dim=1)  # concat => (bs, 1024, h/8, w/8)
        feature = feature * torch.exp(-self.down8(mutual_uncertainty))
        feature = self.conv_bottom(feature)

        # ---- Layer 4 (bottom) ---- #
        seg_list[3] = self.conv_out3(feature)  # (bs, c, h/8, h/8)
        feature, reliable_mask_list[3] = self.UN_C3(feature, seg_list[3],
                                                    self.down8(mutual_evidence),
                                                    self.down8(mutual_uncertainty))  # (512, h/8, w/8)
        feature = self.conv_up3(feature, seg_feature_list[2])  # (512, h/4, w/4)

        # ---- Layer 3 ---- #
        seg_list[2] = self.conv_out2(feature)  # (bs, c, h/4, w/4)
        feature = self.conv_up2(feature, seg_feature_list[1])  # (256, h/2, w/2)

        # ---- Layer 2 ---- #
        seg_list[1] = self.conv_out1(feature)  # (bs, c, h/2, w/2)
        feature = self.conv_up1(feature, seg_feature_list[0])  # (128, h, w)

        # ---- Layer 1 (top) ---- #
        seg_list[0] = self.conv_out0(feature)  # (bs, c, h, w)
        feature, reliable_mask_list[0] = self.UN_C0(feature, seg_list[0],
                                                    mutual_evidence,
                                                    mutual_uncertainty)  # (128, h, w)
        eta = self.down8(feature)

        return seg_list, reliable_mask_list, eta


class ClsHead(nn.Module):
    """
        a sample decoder of classification using conv
    """
    def __init__(self, plus_channels=512):
        super(ClsHead, self).__init__()
        self.cls_head = nn.Sequential(
            SeparableConv2d(1024 + plus_channels, 1024, 3, dilation=2, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            SeparableConv2d(1024, 512, 3, dilation=2, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            SeparableConv2d(512, 512, 3, dilation=2, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            SeparableConv2d(512, 1024, 3, dilation=2, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True))

    def forward(self, cls_feature, mut_feature=None):
        """
        Args:
            cls_feature: the feature of classification after encoding
            mut_feature: the mutual feature

        Returns:
            feature map after decoding (bs, 1024, h/8, w/8)
        """

        if mut_feature is not None:
            x = torch.cat([cls_feature, mut_feature], dim=1)
        else:
            x = cls_feature
        output = self.cls_head(x)

        return output


class UncertaintyInstructorForCls(nn.Module):
    """
        trusted classifier
        refer to TMC
    """

    def __init__(self, cls_classes):
        super(UncertaintyInstructorForCls, self).__init__()

        self.conv_reliable = DoubleConv(in_channels=128, out_channels=1024)
        self.classes = cls_classes
        self.cls_head = ClsHead()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_predict = nn.Sequential(nn.Linear(1024, self.classes, bias=False))

    def forward(self, bottom_cls_feature, bottom_mut_feature, eta):
        """
        Args:
            bottom_cls_feature: bottom cls_feature
            bottom_mut_feature: bottom mutual feature
            eta: feature from UN

        Returns:
            cls_alpha (bs, K) use as the cls_out
            cls_uncertainty (bs, 1) use as the uncertainty
        """

        # ---- extract feature further and get original output ---- #
        bs, c, h, w = bottom_cls_feature.size()
        cls_feats = self.cls_head(bottom_cls_feature, bottom_mut_feature)  # use cls_head to extract feature further
        eta = self.conv_reliable(eta)
        cls_reliable_feature_raw = cls_feats * eta
        cls_feats_reliable = cls_feats + cls_reliable_feature_raw  # concat
        cls_feats_reliable = self.avg_pool(cls_feats_reliable)
        cls_feats_fc = cls_feats_reliable.view(bs, -1)
        cls_out = self.cls_predict(cls_feats_fc)

        # ---- get cls evidence ---- #
        cls_evidence = self.infer(cls_out)  # NOTE : use the last layer output as evidence also we use it as cls result

        # ---- use dirichlet distribution ---- #
        cls_alpha = cls_evidence + 1  # (bs, 2)

        # ---- get belief and uncertainty ---- #
        S = torch.sum(cls_alpha, dim=1, keepdim=True)  # must keep dim keep it as (bs, 1) not (bs,)
        cls_belief = (cls_alpha - 1) / (S.expand(cls_alpha.shape))  # belief (bs, 2) case-wise
        cls_uncertainty = self.classes / S  # uncertainty (bs, 1) case-wise

        return cls_alpha, cls_uncertainty

    def infer(self, input):
        evidence = F.softplus(input)
        return evidence
