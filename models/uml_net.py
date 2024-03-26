# -*- coding: utf-8 -*-
# @Time    : 2023/3/21 16:01
# @Author  : Karry Ren

""" The Uncertainty Mutual Leaning Neural Network.

The network is little different from our paper. This network is more powerful !

You might have some questions about our Network:
    1. Why we choose Res2Net as a feature encoder backbone ?
        To be honest, at the beginning of this study, we ref one paper to build up our mutual learning network `JCS`
          Ref. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9357961
        That paper selected Res2Net as the classification backbone and VGG16 as the segmentation backbone.
          As you can see, in our released MICCAI2023 paper we build up our UML_Net just like it.
        But in the following experiments, we found that using Res2Net to do the classification and segmentation is
          more accurate and robust, so we choose Res2Net as the backbone for both cls and seg feature encoding !
          They have the same structure while not sharing the params.

    2. The eta is changed ?
        Yes, the eta is from mutual features rather than segmentation features to classification.

    3. No back information from mutual feature to classification and segmentation ?
        Yes, each feature list used for each task.

    4. UN is changed ?
        Yes, we use a new Attention-U structure rather than the concat operation to put the mutual info and uncertainty
          to segmentation feature.

"""

from typing import Optional
import torch
import torch.nn as nn

from models.model_lib.res2net import res2net50_v1b_26w_4s
from models.modules import PairWiseFeatureMixer, MutualFeatureDecoder
from models.modules import UNSegDecoder
from models.modules import UIClsDecoder


class UML_Net(nn.Module):
    """ Uncertainty Mutual Learning Neural Network. """

    def __init__(self, pretrained_res2net_path: Optional[str], seg_class: int, cls_class: int):
        """ Init function of UML_Net. Here we will build up all modules which
            can be divided into 3 parts:
                - Part 1. The two pretrained feature encoders for cls and seg (both are Res2Net).
                - Part 2. The mutual feature mixer and decoder.
                - Part 3. The un_seg_decoder.

        :param pretrained_res2net_path: the pretrained Res2Net path
        :param seg_class: the total class of segmentation
        :param cls_class: the total class of classification

        """

        super(UML_Net, self).__init__()

        # ---- Part 2. The mutual feature mixer and decoder ---- #
        self.mut_feature_mixer = PairWiseFeatureMixer()  # the mutual feature mixer
        self.mut_feature_decoder = MutualFeatureDecoder(seg_class=seg_class)  # the mutual feature decoder
        self.mut_up_sample_dict = nn.ModuleDict({
            "2": nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            "4": nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            "8": nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        })  # the up_sample dict for mutual result, must use nn.Dict or backward wrong !

        # ---- Part 3. The un_seg_decoder ---- #
        self.un_seg_decoder = UNSegDecoder(seg_class=seg_class)

        # ---- Part 4. The un_cls_decoder ---- #
        self.un_cls_decoder = UIClsDecoder(cls_class=cls_class)

        # ---- Init the params of all modules ---- #
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # The 2d Conv layer using KaiMing normal
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):  # The BN use constant 1 for weight and 0 for bias
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # ---- Part 1. The two pretrained feature encoders (both are Res2Net) ---- #
        # because the pretraining, must init the two feature encoders after param !!!
        self.cls_feature_encoder = res2net50_v1b_26w_4s(pretrained=pretrained_res2net_path)  # classification
        self.seg_feature_encoder = res2net50_v1b_26w_4s(pretrained=pretrained_res2net_path)  # segmentation

    def forward(self, x: torch.Tensor):
        """ Forward function of UML_Net.

        :param x: the input image (bs, c, h, w)

        The detail process is as follows:
            Step 1. Feature Encoding to get `cls_feature_list` and `seg_feature_list`
                Because the encoders of classification and segmentation are both Res2Net. The feature_list
                  of classification and segmentation have the same shape items:
                    - item_0, shape=(bs, 64, h, w)
                    - item_1, shape=(bs, 256, h/2, w/2)
                    - item_2, shape=(bs, 521, h/4, w/4)
                    - item_3, shape=(bs, 1024, h/8, w/8)
            Step 2. Mutual Feature Mixing and Decoding to get the `mut_feature_list` and `decoding_features`.
                The `mut_feature_list`  has the same shape items with cls and seg feature list.
                The `decoding_features` include:
                  - mut_evidence_list: 4 layer mutual evidence, shape=(bs, seg_class, h/2^{i-1}, w/2^{i-1})
                  - mut_alpha_list: 4 layer mutual alpha, shape=(bs, seg_class, h/2^{i-1}, w/2^{i-1})
                  - mut_uncertainty_list: 4 layer mutual alpha, shape=(bs, 1, h/2^{i-1}, w/2^{i-1})
                  - mut_info_list:
                    ~ item_0, shape=(bs, 64, h, w)
                    ~ item_1, shape=(bs, 128, h/2, w/2)
                    ~ item_2, shape=(bs, 256, h/4, w/4)
                    ~ item_3, shape=(bs, 512, h/8, w/8)
                For the deep supervision, we need to do the UpSampling to get:
                  - mut_evidence_list: 4 layer mutual evidence (bs, seg_class, h, w)
                  - mut_alpha_list: 4 layer mutual alpha (bs, seg_class, h, w)
                  - mut_uncertainty_list: 4 layer mutual alpha (bs, 1, h, w)
            Step 3. Do the Segmentation use `seg_feature_list` guiding by  `mut_info_list` and
                    `top pixel-wise uncertainty` to get the `final_seg`, shape=(bs, seg_class, h, w),
                     while generating the `eta_for_cls` to guide the UI, shape=(bs, 64, h/8, w/8)
            Step 4. Do the Classification use `cls_feature` guiding by the `eta_for_cls`
                    to get the `cls_alpha`, shape=(bs, cls_class) and `cls_uncertainty`, shape=(bs, 1)

        returns:
            - cls_alpha: the final classification result, shape=(bs, cls_class)
            - cls_uncertainty: the image-level uncertainty, shape=(bs, 1)
            - mut_evidence_list: 4 layer mutual evidence, shape=(bs, seg_class, h/2^{i-1}, w/2^{i-1})
            - mut_alpha_list: 4 layer mutual alpha, shape=(bs, seg_class, h/2^{i-1}, w/2^{i-1})
            - mut_uncertainty_list: 4 layer mutual alpha, shape=(bs, 1, h/2^{i-1}, w/2^{i-1})
            - final_seg: the final segmentation result, shape=(bs, seg_class, h, w)

        """

        # ---- Step 1. Feature Encoding ---- #
        cls_feature_list = self.cls_feature_encoder(x)
        seg_feature_list = self.seg_feature_encoder(x)

        # ---- Step 2. Mutual Feature Mixing & Decoding and UpSample for deep Supervision ---- #
        mut_feature_list = self.mut_feature_mixer(cls_feature_list, seg_feature_list)
        (mut_evidence_list, mut_alpha_list, mut_uncertainty_list,
         mut_info_list) = self.mut_feature_decoder(mut_feature_list)
        for i in range(1, len(mut_evidence_list)):  # layer 2 to 4
            mut_evidence_list[i] = self.mut_up_sample_dict[str(2 ** i)](mut_evidence_list[i])
            mut_alpha_list[i] = self.mut_up_sample_dict[str(2 ** i)](mut_alpha_list[i])
            mut_uncertainty_list[i] = self.mut_up_sample_dict[str(2 ** i)](mut_uncertainty_list[i])

        # ---- Step 3. Do the Segmentation ---- #
        un_mut_uncertainty = mut_uncertainty_list[0]
        final_seg, eta_for_cls = self.un_seg_decoder(seg_feature_list, mut_info_list, un_mut_uncertainty)

        # ---- Step 4. Do the Classification ---- #
        cls_feature = cls_feature_list[-1]
        cls_alpha, cls_uncertainty = self.un_cls_decoder(cls_feature, eta_for_cls)

        return cls_alpha, cls_uncertainty, mut_evidence_list, mut_alpha_list, mut_uncertainty_list, final_seg


if __name__ == "__main__":  # A demo using UML_Net
    # ---- Step 1. Build the UML_Net ---- #
    model = UML_Net(pretrained_res2net_path=None, seg_class=2, cls_class=2)

    # ---- Step 2. Compute the total params ---- #
    total_param = sum([param.nelement() for param in model.parameters()])
    print(f"Number of model's parameter: {total_param / 1e6} M")

    # ---- Step 3. Forward ---- #
    images = torch.rand(1, 3, 256, 256)
    ca, cu, mel, mal, mul, fs = model(images)
    print("cls_alpha: ")
    print(ca.shape)
    print("cls_uncertainty: ")
    print(cu.shape)
    print("mutual_evidence_list: ")
    for m_e in mel:
        print(m_e.shape)
    print("mut_alpha_list: ")
    for m_a in mal:
        print(m_a.shape)
    print("mut_uncertainty_list: ")
    for m_u in mul:
        print(m_u.shape)
    print("final_seg: ")
    print(fs.shape)
