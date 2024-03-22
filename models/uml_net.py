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

"""

from typing import Optional
import torch
import torch.nn as nn

from model_lib.res2net import res2net50_v1b_26w_4s
from modules import PairWiseFeatureMixer, MutualFeatureDecoder


class UML_Net(nn.Module):
    """ Uncertainty Mutual Learning Neural Network. """

    def __init__(self, pretrained_res2net_path: Optional[str], seg_class: int):
        """ Init function of UML_Net. Here we will build up all modules which
            can be divided into 3 parts:
                - Part 1. The two pretrained feature encoders for cls and seg (both are Res2Net).
                - Part 2. The mutual feature mixer and decoder.

        :param pretrained_res2net_path: the pretrained Res2Net path
        :param seg_class: the total class of segmentation

        """

        super(UML_Net, self).__init__()

        # ---- Part 2. The mutual feature mixer and decoder ---- #
        self.mut_feature_mixer = PairWiseFeatureMixer()  # the mutual feature mixer
        self.mut_feature_decoder = MutualFeatureDecoder(seg_class=seg_class)

        # - init_params
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
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
                  - mut_evidence_list: 4 layer mutual evidence (bs, seg_class, h/2^{i-1}, w/2^{i-1})
                  - mut_alpha_list: 4 layer mutual alpha (bs, seg_class, h/2^{i-1}, w/2^{i-1})
                  - mut_uncertainty_list: 4 layer mutual alpha (bs, 1, h/2^{i-1}, w/2^{i-1})
                  - mut_info_list: [(bs, 64, h, w), (bs, 128, h/2, w/2), (bs, 256, h/4, w/4)]



        returns:
            cls_alpha: (bs, cls_cls)
            cls_uncertainty: (bs, 1)
            mutual_evidence: (bs, seg_cls, h, w)
            mutual_alpha: (bs, seg_cls, h, w)
            mutual_uncertainty: (bs, 1, h, w)
            seg_list: four seg results having same size (bs, seg_cls, h, w)
            seg_list_view: four seg results having different size

        """

        # ---- Step 1. Feature Encoding ---- #
        cls_feature_list = self.cls_feature_encoder(x)
        seg_feature_list = self.seg_feature_encoder(x)

        # ---- Step 2. Mutual Feature Mixing and Decoding ---- #
        mut_feature_list = self.mut_feature_mixer(cls_feature_list, seg_feature_list)
        (mut_evidence_list, mut_alpha_list, mut_uncertainty_list,
         mut_info_list, eta_for_cls) = self.mut_feature_decoder(mut_feature_list)

        return None


if __name__ == "__main__":  # A demo using UML_Net

    # ---- Step 1. Build the UML_Net ---- #
    model = UML_Net(pretrained_res2net_path=None, seg_class=3)

    # ---- Step 2. Compute the total params ---- #
    total_param = sum([param.nelement() for param in model.parameters()])
    print(f"Number of model's parameter: {total_param / 1e6} M")

    # ---- Step 3. Forward ---- #
    images = torch.rand(1, 3, 256, 256)
    a = model(images)
