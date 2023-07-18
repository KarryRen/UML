from models.lib.res2net import res2net50_v1b_26w_4s
from models.lib.vgg import vgg16
from models.modules import *


class UMLNet(nn.Module):
    """
        has Mutual feature Decoder and UN and UI
        Two Encoder final network
    """

    def __init__(self, config):
        super(UMLNet, self).__init__()

        self.feature_mixer = PairWiseFeatureMixer()
        self.mutual_feature_decoder = TrustedMutualFeatureDecoder(seg_classes=config.NUM_CLASSES_SEG)

        self.UN = UncertaintyNavigatorForSeg(seg_classes=config.NUM_CLASSES_SEG)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.UI = UncertaintyInstructorForCls(cls_classes=config.NUM_CLASSES_CLS)

        # ---- init_params ---- #
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # ---- if pre-train must set them after init_params !!! ---- #
        self.cls_feature_encoder = res2net50_v1b_26w_4s(pretrained=config.PRETRAIN_RES2NET)
        self.seg_feature_encoder = vgg16(pretrained=config.PRETRAIN_VGG)

    def forward(self, x):
        # ---- Mutual Feature Learning ---- #
        cls_feature_list = self.cls_feature_encoder(
            x)  # cfl : [(64, h, w), (256, h/2, w/2), (512, h/4, w/4), (1024, h/8, w/8)]
        seg_feature_list = self.seg_feature_encoder(
            x)  # sfl : [(64, h, w), (128, h/2, w/2), (256, h/4, w/4), (512, h/8, w/8)]
        mutual_feature_list = self.feature_mixer(
            cls_feature_list=cls_feature_list,
            seg_feature_list=seg_feature_list)  # jfl : [(64, h, w), (128, h/2, w/2), (256, h/4, w/4), (512, h/8, w/8)]

        bottom_mutual_feature = mutual_feature_list[-1]  # the bottom feature of jfl [512, h/8, w/8]

        # ---- Mutual Feature Decoding ---- #
        mutual_evidence, mutual_alpha, mutual_uncertainty = self.mutual_feature_decoder(mutual_feature_list)

        # ---- Uncertainty Navigator For Segmentation  ----#
        seg_list, reliable_mask_list, eta = self.UN(seg_feature_list=seg_feature_list,
                                                    bottom_mut_feature=bottom_mutual_feature,
                                                    mutual_evidence=mutual_evidence,
                                                    mutual_uncertainty=mutual_uncertainty)
        seg_list_view = seg_list.copy()
        seg_list[1] = self.up2(seg_list[1])
        seg_list[2] = self.up4(seg_list[2])
        seg_list[3] = self.up8(seg_list[3])

        # ---- Uncertainty Instructor for Classification ---- #
        bottom_cls_feature = cls_feature_list[-1]
        cls_alpha, cls_uncertainty = self.UI(bottom_cls_feature=bottom_cls_feature,
                                             bottom_mut_feature=bottom_mutual_feature,
                                             eta=eta)

        return cls_alpha, cls_uncertainty, \
               mutual_evidence, mutual_alpha, mutual_uncertainty, \
               seg_list, seg_list_view, reliable_mask_list
