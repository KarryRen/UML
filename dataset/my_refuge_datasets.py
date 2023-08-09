import torch.utils.data
import os
import config as config
from utils.visualization import *


def file_extend(PATH_root, file_name_list):
    extend_file_name_list = []
    for i in range(len(file_name_list)):
        extend_file_name_list.append(PATH_root + "\\" + file_name_list[i])
    return extend_file_name_list


class JointRefugeDataset(torch.utils.data.Dataset):
    """ get refuge data """

    def __init__(self, path_refuge_data, data_type='train'):
        self.data_type = data_type

        # ---- img and msk filename list ---- #
        self.img_list = []
        self.img_list_0 = []
        self.img_list_1 = []
        self.msk_list = []

        # - step 1 get image and mask file folder name
        img_folder_name = path_refuge_data + '\\' + data_type + '\\images'
        msk_folder_name = path_refuge_data + '\\' + data_type + '\\masks'

        # - step 2 get image and mask file name
        img_name_list = os.listdir(img_folder_name)
        msk_name_list = os.listdir(msk_folder_name)

        # - step 3 get full name of image and mask
        img_full_name_list = file_extend(img_folder_name, img_name_list)
        msk_full_name_list = file_extend(msk_folder_name, msk_name_list)
        assert len(img_full_name_list) == len(msk_full_name_list)

        # - step 4 extend path of images and masks
        self.img_list.extend(img_full_name_list)
        self.msk_list.extend(msk_full_name_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = plt.imread(self.img_list[idx]).copy()
        mask = plt.imread(self.msk_list[idx]).copy()
        label = int(self.img_list[idx].split("_")[-2])
        image_name = self.img_list[idx].split("\\")[-1]
        image = image.transpose((2, 0, 1)).astype(np.float32)  # change data type to FloatTensor
        # mask = mask[:, :, 0]

        # - must do to avoid pixel != 0 or != 1
        mask[mask < 10] = 2
        mask[(mask >= 10) & (mask <= 245)] = 1
        mask[mask > 245] = 0
        mask = mask.astype(np.int64)  # # change data type to LongTensor

        return image / 255, mask, label, image_name

    def get_img_info(self, idx):
        image = plt.imread(self.img_list[idx])
        return {"height": image.shape[0], "width": image.shape[1]}
