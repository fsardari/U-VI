import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
from PIL import Image, ImageFilter
import pickle
import matplotlib.pyplot as plt


class NTU_Dense(Dataset):

    def __init__(self, xlsx_file_name, annotation_dir, densepose_dir, orginal_img_size=512, final_img_size=128):

        self.transform = transforms.ToTensor()
        self.annotation_dir = annotation_dir
        self.densepose_dir = densepose_dir
        self.final_img_size = final_img_size
        self.orginal_img_size = orginal_img_size
        self.file = pd.read_csv(os.path.join(annotation_dir, xlsx_file_name))

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):

        a_sheet = self.file.iloc
        # total_heatmap_numbers = 26
        # margin = 10
        ra_f_idx = np.random.randint(1, 100)

        a_setup_name = a_sheet[idx, 0]
        a_action_name = a_sheet[idx, 1]
        a_person_name = a_sheet[idx, 2]
        a_rr_action_name = a_sheet[idx, 3]
        self.rr_action_name = a_rr_action_name
        # a_device_number = a_sheet[idx, 4]
        r_device = np.random.randint(1, 3)
        if r_device == 2:
            a_device_number = 'device_1'
            b_device_number = 'device_2'
            a_to_b_rot = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
            b_to_a_rot = torch.tensor([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        else:
            a_device_number = 'device_2'
            b_device_number = 'device_1'
            a_to_b_rot = torch.tensor([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
            b_to_a_rot = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

        c_device_number = 'device_0'
        a_img_name = a_sheet[idx, 6]
        sub_folder = 'COLOR'

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% three simultaneous frames from three different views %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        root_dense_path = self.densepose_dir + '/' + a_setup_name + '/' + a_action_name

        a_dense_path = root_dense_path + '/' + a_person_name + '/' + a_rr_action_name + '/' + a_device_number + '/' + sub_folder
        b_dense_path = root_dense_path + '/' + a_person_name + '/' + a_rr_action_name + '/' + b_device_number + '/' + sub_folder
        c_dense_path = root_dense_path + '/' + a_person_name + '/' + a_rr_action_name + '/' + c_device_number + '/' + sub_folder

        general_dense = torch.zeros([1, 3, self.orginal_img_size, self.orginal_img_size])

        [a_dense_img, a_bbox, a_w, a_h] = self.load_dense_data(a_dense_path, a_img_name)
        [b_dense_img, b_bbox, b_w, b_h] = self.load_dense_data(b_dense_path, a_img_name)
        [c_dense_img, c_bbox, c_w, c_h] = self.load_dense_data(c_dense_path, a_img_name)

        general_dense[0] = a_dense_img
        a_dense_org_img = a_dense_img
        # a_dense_img = F.interpolate(general_dense, size=self.final_img_size, mode='area')[0]
        a_dense_img = F.interpolate(general_dense, size=self.final_img_size)[0]

        general_dense[0] = b_dense_img
        b_dense_org_img = b_dense_img
        # b_dense_img = F.interpolate(general_dense, size=self.final_img_size, mode='area')[0]
        b_dense_img = F.interpolate(general_dense, size=self.final_img_size)[0]
        general_dense[0] = c_dense_img
        c_dense_img = F.interpolate(general_dense, size=self.final_img_size)[0]
        # c_dense_img = F.interpolate(general_dense, size=self.final_img_size, mode='area')[0]

        ################################ data augmentation ############################################################

        [a_dense_aug_img, a_axis, a_aug_pixel] = self.data_augmentation(a_dense_org_img, a_device_number, a_bbox, a_w, a_h)
        general_dense[0] = a_dense_aug_img
        a_dense_aug_img = F.interpolate(general_dense, size=self.final_img_size)[0]
        # a_dense_aug_img = F.interpolate(general_dense, size=self.final_img_size, mode='area')[0]

        [b_dense_aug_img, b_axis, b_aug_pixel] = self.data_augmentation(b_dense_org_img, b_device_number, b_bbox, b_w, b_h)
        general_dense[0] = b_dense_aug_img
        # b_dense_aug_img = F.interpolate(general_dense, size=self.final_img_size, mode='area')[0]
        b_dense_aug_img = F.interpolate(general_dense, size=self.final_img_size)[0]
        ################################   a2 and b2 data   ###################################################################

        [ra_f_idx, a2_img_name, a2_person_name, a2_rr_action_name] = self.random_distance_frame(idx, ra_f_idx, a_person_name, a_rr_action_name)
        d = ra_f_idx
        a2_dense_path = root_dense_path + '/' + a2_person_name + '/' + a2_rr_action_name + '/' + a_device_number + '/' + sub_folder
        b2_dense_path = root_dense_path + '/' + a2_person_name + '/' + a2_rr_action_name + '/' + b_device_number + '/' + sub_folder

        [a2_dense_img, a2_bbox, a2_w, a2_h] = self.load_dense_data2(a2_dense_path, a2_img_name)
        [b2_dense_img, b2_bbox, b2_w, b2_h] = self.load_dense_data2(b2_dense_path, a2_img_name)

        general_dense[0] = a2_dense_img
        a2_dense_img = F.interpolate(general_dense, size=self.final_img_size)[0]
        # a2_dense_img = F.interpolate(general_dense, size=self.final_img_size, mode='area')[0]
        general_dense[0] = b2_dense_img
        b2_dense_img = F.interpolate(general_dense, size=self.final_img_size)[0]
        # b2_dense_img = F.interpolate(general_dense, size=self.final_img_size, mode='area')[0]

        ######################

        r_f = np.random.randint(1, 3)
        if r_f == 2:
            a_dense_img = torch.flip(a_dense_img, [2])
            b_dense_img = torch.flip(b_dense_img, [2])
            c_dense_img = torch.flip(c_dense_img, [2])

            a2_dense_img = torch.flip(a2_dense_img, [2])
            b2_dense_img = torch.flip(b2_dense_img, [2])

            a_dense_aug_img = torch.flip(a_dense_aug_img, [2])
            if a_axis != 'y':
                a_aug_pixel = -1 * a_aug_pixel
            b_dense_aug_img = torch.flip(b_dense_aug_img, [2])
            if b_axis != 'y':
                b_aug_pixel = -1 * b_aug_pixel

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% color transformations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        return {'A_Dense': a_dense_img, 'A_Dense_aug': a_dense_aug_img, 'a_axis': a_axis, 'a_aug_pixel': a_aug_pixel, 'a_b_rot': a_to_b_rot, 'A_path': a_dense_path + '/' + a_img_name + '.png',
                'B_Dense': b_dense_img, 'B_Dense_aug': b_dense_aug_img, 'b_axis': b_axis,  'b_aug_pixel': b_aug_pixel, 'b_a_rot': b_to_a_rot, 'B_path': b_dense_path + '/' + a_img_name + '.png',
                'C_Dense': c_dense_img, 'A2_Dense': a2_dense_img, 'B2_Dense': b2_dense_img, 'd': d}

    def data_augmentation(self, dense_org_img, device_number, bbox, w, h):

        max_r = 50
        dense_aug_img = torch.zeros([3, self.orginal_img_size, self.orginal_img_size])
        r_v_h = np.random.randint(1, 3)
        if r_v_h == 1:
            # vertical augmentation
            r_s = np.random.randint(1, 3)
            axis = 'y'
            if r_s == 1:
                # up side aumentation
                try:
                    if bbox[1] > max_r:
                        p_s = np.random.randint(1, max_r)
                    else:
                        p_s = np.random.randint(1, bbox[1])
                except:
                    p_s = 1
                aug_pixel = (-1 * p_s)/(self.orginal_img_size/self.final_img_size)
                dense_aug_img[:, 0: self.orginal_img_size - p_s, :] = dense_org_img[:, p_s:self.orginal_img_size, :]
            else:
                # down side aumentation
                try:
                    if self.orginal_img_size - (bbox[1] + h) > max_r:
                        p_s = np.random.randint(1, max_r)
                    else:
                        p_s = np.random.randint(1, self.orginal_img_size - (bbox[1] + h))
                except:
                    p_s = 1
                aug_pixel = p_s/(self.orginal_img_size/self.final_img_size)
                dense_aug_img[:, p_s: self.orginal_img_size,:] = dense_org_img[:, 0: self.orginal_img_size - p_s,:]
        else:
            # if device_number == 'device_1':
            #     axis = 'x'
            # else:
            #     axis = 'z'
            axis = 'x'
            r_s = np.random.randint(1, 3)
            if r_s == 1:
                # left side aumentation
                try:
                    if bbox[0] > max_r:
                        p_s = np.random.randint(1, max_r)
                    else:
                        p_s = np.random.randint(1, bbox[0])
                except:
                    p_s = 1
                aug_pixel = (-1 * p_s)/(self.orginal_img_size/self.final_img_size)
                dense_aug_img[:, :, 0: self.orginal_img_size - p_s] = dense_org_img[:, :, p_s:self.orginal_img_size]
            else:
                # right side aumentation
                try:
                    if self.orginal_img_size - (bbox[0] + w) > max_r:
                        p_s = np.random.randint(1, max_r)
                    else:
                        p_s = np.random.randint(1, self.orginal_img_size - (bbox[0] + w))
                except:
                    p_s = 1
                aug_pixel = p_s/(self.orginal_img_size/self.final_img_size)
                dense_aug_img[:, :, p_s: self.orginal_img_size] = dense_org_img[:, :, 0: self.orginal_img_size - p_s]

        return dense_aug_img, axis, aug_pixel

    def load_dense_data(self, a_dense_path, a_img_name):

        a_f = open(a_dense_path + '/' + a_img_name + '.pkl', 'rb')
        a_dense_img = torch.zeros([3, self.orginal_img_size, self.orginal_img_size])
        a_data = pickle.load(a_f)

        for i in range(len(a_data)-1):
            a_bbox = a_data[i][0].int()
            a_ch, a_h, a_w = a_data[i][1].size()
            a_dense_img[:, a_bbox[1]:a_bbox[1] + a_h, a_bbox[0]:a_bbox[0] + a_w] = a_data[i][1]
            if i >= 2:
                a_bbox[0] = torch.min(a_bbox[0], a_data[i-1][0].int())
                a_bbox[1] = torch.min(a_bbox[1], a_data[i-1][1].int())

                a_w = torch.max(a_bbox[0], a_data[i-1][0].int()) - a_bbox[0] + 1
                a_h = torch.max(a_bbox[1], a_data[i-1][1].int()) - a_bbox[1] + 1

        a_dense_img[0, :, :] = a_dense_img[0, :, :] / 24
        a_dense_img[1, :, :] = a_dense_img[1, :, :] / 255
        a_dense_img[2, :, :] = a_dense_img[2, :, :] / 255

        return a_dense_img, a_bbox, a_w, a_h

    def load_dense_data2(self, a_dense_path, a_img_name):

        a_f = open(a_dense_path + '/' + a_img_name + '.pkl', 'rb')
        a_dense_img = torch.zeros([3, self.orginal_img_size, self.orginal_img_size])
        a_data = pickle.load(a_f)

        for i in range(len(a_data)-1):
            a_bbox = a_data[i][0].int()
            a_ch, a_h, a_w = a_data[i][1].size()
            a_dense_img[:, a_bbox[1]:a_bbox[1] + a_h, a_bbox[0]:a_bbox[0] + a_w] = a_data[i][1]

        a_dense_img[0, :, :] = a_dense_img[0, :, :] / 24
        a_dense_img[1, :, :] = a_dense_img[1, :, :] / 255
        a_dense_img[2, :, :] = a_dense_img[2, :, :] / 255

        return a_dense_img, a_bbox, a_w, a_h

    def random_distance_frame(self, idx, ra_f_idx, a_person_name, a_rr_action_name):
        a_sheet = self.file.iloc
        flag = True
        i = 1
        while flag:
            if idx + ra_f_idx < len(self.file):
                a2_person_name = a_sheet[idx + ra_f_idx, 2]
                a2_rr_action_name = a_sheet[idx + ra_f_idx, 3]
                a2_img_name = a_sheet[idx + ra_f_idx, 6]
                if a2_person_name != a_person_name or a2_rr_action_name != a_rr_action_name:
                    ra_f_idx = -1 * ra_f_idx
                    if i % 2 == 0:
                        ra_f_idx = ra_f_idx//2
                else:
                    flag = False
                i = i + 1
            else:
                ra_f_idx = len(self.file) - idx - 1

        return ra_f_idx, a2_img_name, a2_person_name, a2_rr_action_name


class NTU_Depth(Dataset):

    def __init__(self, xlsx_file_name, annotation_dir, densepose_dir, depth_dir, orginal_img_size=512, final_img_size=128):

        self.transform = transforms.ToTensor()
        self.annotation_dir = annotation_dir
        self.densepose_dir = densepose_dir
        self.depth_dir = depth_dir
        self.final_img_size = final_img_size
        self.orginal_img_size = orginal_img_size
        self.file = pd.read_csv(os.path.join(annotation_dir, xlsx_file_name))
        self.filter = ImageFilter.MaxFilter(size=3)

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):

        a_sheet = self.file.iloc
        # total_heatmap_numbers = 26
        # margin = 10
        ra_f_idx = np.random.randint(1, 100)

        a_setup_name = a_sheet[idx, 0]
        a_action_name = a_sheet[idx, 1]
        a_person_name = a_sheet[idx, 2]
        a_rr_action_name = a_sheet[idx, 3]
        self.rr_action_name = a_rr_action_name
        # a_device_number = a_sheet[idx, 4]
        r_device = np.random.randint(1, 3)
        if r_device == 2:
            a_device_number = 'device_1'
            b_device_number = 'device_2'
            a_to_b_rot = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
            b_to_a_rot = torch.tensor([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        else:
            a_device_number = 'device_2'
            b_device_number = 'device_1'
            a_to_b_rot = torch.tensor([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
            b_to_a_rot = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

        c_device_number = 'device_0'
        a_img_name = a_sheet[idx, 6]
        sub_folder = 'COLOR'

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% three simultaneous frames from three different views %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        root_dense_path = self.densepose_dir + '/' + a_setup_name + '/' + a_action_name
        root_depth_path = self.depth_dir + '/' + 'S' + a_setup_name[1:].zfill(3)

        a_dense_path = root_dense_path + '/' + a_person_name + '/' + a_rr_action_name + '/' + a_device_number + '/' + sub_folder
        b_dense_path = root_dense_path + '/' + a_person_name + '/' + a_rr_action_name + '/' + b_device_number + '/' + sub_folder
        c_dense_path = root_dense_path + '/' + a_person_name + '/' + a_rr_action_name + '/' + c_device_number + '/' + sub_folder

        a_depth_path = root_depth_path + 'C' + str(int(a_device_number[7]) + 1).zfill(3) + 'P' + a_person_name[1:].zfill(3) + 'R' + a_rr_action_name[1:].zfill(3) + 'A' + a_action_name[1:].zfill(3)
        b_depth_path = root_depth_path + 'C' + str(int(b_device_number[7]) + 1).zfill(3) + 'P' + a_person_name[1:].zfill(3) + 'R' + a_rr_action_name[1:].zfill(3) + 'A' + a_action_name[1:].zfill(3)
        c_depth_path = root_depth_path + 'C' + str(int(c_device_number[7]) + 1).zfill(3) + 'P' + a_person_name[1:].zfill(3) + 'R' + a_rr_action_name[1:].zfill(3) + 'A' + a_action_name[1:].zfill(3)

        general_dense = torch.zeros([1, 3, self.orginal_img_size, self.orginal_img_size])

        [a_depth_img, a_bbox, a_w, a_h] = self.load_depth_data(a_dense_path, a_depth_path, a_img_name)
        [b_depth_img, b_bbox, b_w, b_h] = self.load_depth_data(b_dense_path, b_depth_path, a_img_name)
        [c_depth_img, c_bbox, c_w, c_h] = self.load_depth_data(c_dense_path, c_depth_path, a_img_name)

        general_dense[0] = a_depth_img
        a_depth_org_img = a_depth_img
        a_depth_img = F.interpolate(general_dense, size=self.final_img_size, mode='area')[0]

        general_dense[0] = b_depth_img
        b_depth_org_img = b_depth_img
        b_depth_img = F.interpolate(general_dense, size=self.final_img_size, mode='area')[0]

        general_dense[0] = c_depth_img
        c_depth_img = F.interpolate(general_dense, size=self.final_img_size, mode='area')[0]

        ################################ data augmentation ############################################################


        [a_depth_aug_img, a_axis, a_aug_pixel] = self.data_augmentation(a_depth_org_img, a_bbox, a_w, a_h)
        general_dense[0] = a_depth_aug_img
        a_depth_aug_img = F.interpolate(general_dense, size=self.final_img_size, mode='area')[0]

        [b_depth_aug_img, b_axis, b_aug_pixel] = self.data_augmentation(b_depth_org_img, b_bbox, b_w, b_h)
        general_dense[0] = b_depth_aug_img
        b_depth_aug_img = F.interpolate(general_dense, size=self.final_img_size, mode='area')[0]

        ################################   a2 and b2 data   ###################################################################

        [ra_f_idx, a2_img_name, a2_person_name, a2_rr_action_name] = self.random_distance_frame(idx, ra_f_idx, a_person_name, a_rr_action_name)
        d = ra_f_idx
        a2_dense_path = root_dense_path + '/' + a2_person_name + '/' + a2_rr_action_name + '/' + a_device_number + '/' + sub_folder
        b2_dense_path = root_dense_path + '/' + a2_person_name + '/' + a2_rr_action_name + '/' + b_device_number + '/' + sub_folder

        a2_depth_path = root_depth_path + 'C' + str(int(a_device_number[7]) + 1).zfill(3) + 'P' + a2_person_name[1:].zfill(3) + 'R' + a2_rr_action_name[1:].zfill(3) + 'A' + a_action_name[1:].zfill(3)
        b2_depth_path = root_depth_path + 'C' + str(int(b_device_number[7]) + 1).zfill(3) + 'P' + a2_person_name[1:].zfill(3) + 'R' + a2_rr_action_name[1:].zfill(3) + 'A' + a_action_name[1:].zfill(3)

        [a2_depth_img, a2_bbox, a2_w, a2_h] = self.load_depth_data2(a2_dense_path, a2_depth_path, a2_img_name)
        [b2_depth_img, b2_bbox, b2_w, b2_h] = self.load_depth_data2(b2_dense_path, b2_depth_path, a2_img_name)

        general_dense[0] = a2_depth_img
        a2_depth_img = F.interpolate(general_dense, size=self.final_img_size, mode='area')[0]
        general_dense[0] = b2_depth_img
        b2_depth_img = F.interpolate(general_dense, size=self.final_img_size, mode='area')[0]

        ######################

        r_f = np.random.randint(1, 3)
        if r_f == 2:
            a_depth_img = torch.flip(a_depth_img, [2])
            b_depth_img = torch.flip(b_depth_img, [2])
            c_depth_img = torch.flip(c_depth_img, [2])

            a2_depth_img = torch.flip(a2_depth_img, [2])
            b2_depth_img = torch.flip(b2_depth_img, [2])

            a_depth_aug_img = torch.flip(a_depth_aug_img, [2])
            if a_axis != 'y':
                a_aug_pixel = -1 * a_aug_pixel
            b_depth_aug_img = torch.flip(b_depth_aug_img, [2])
            if b_axis != 'y':
                b_aug_pixel = -1 * b_aug_pixel

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% color transformations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        return {'A_Dense': self.change_view(a_depth_img), 'A_Dense_aug': self.change_view(a_depth_aug_img), 'a_axis': a_axis, 'a_aug_pixel': a_aug_pixel, 'a_b_rot': a_to_b_rot, 'A_path': a_depth_path + '/' + a_img_name + '.png',
                'B_Dense': self.change_view(b_depth_img), 'B_Dense_aug': self.change_view(b_depth_aug_img), 'b_axis': b_axis,  'b_aug_pixel': b_aug_pixel, 'b_a_rot': b_to_a_rot, 'B_path': b_dense_path + '/' + a_img_name + '.png',
                'C_Dense': self.change_view(c_depth_img), 'A2_Dense': self.change_view(a2_depth_img), 'B2_Dense': self.change_view(b2_depth_img), 'd': d}

    def data_augmentation(self, depth_org_img, bbox, w, h):

        max_r = 100
        depth_aug_img = torch.zeros([3, self.orginal_img_size, self.orginal_img_size])
        r_v_h = np.random.randint(1, 3)
        if r_v_h == 1:
            # vertical augmentation
            r_s = np.random.randint(1, 3)
            axis = 'y'
            if r_s == 1:
                # up side aumentation
                try:
                    if bbox[1] > max_r:
                        p_s = np.random.randint(1, max_r)
                    else:
                        p_s = np.random.randint(1, bbox[1])
                except:
                    p_s = 1
                aug_pixel = (-1 * p_s)/(self.orginal_img_size/self.final_img_size)
                depth_aug_img[:, 0: self.orginal_img_size - p_s, :] = depth_org_img[:, p_s:self.orginal_img_size, :]
                depth_aug_img[:, self.orginal_img_size - p_s: self.orginal_img_size, :] = depth_org_img[:, 0:p_s , :]
            else:
                # down side aumentation
                try:
                    if self.orginal_img_size - (bbox[1] + h) > max_r:
                        p_s = np.random.randint(1, max_r)
                    else:
                        p_s = np.random.randint(1, self.orginal_img_size - (bbox[1] + h))
                except:
                    p_s = 1
                aug_pixel = p_s/(self.orginal_img_size/self.final_img_size)
                depth_aug_img[:, p_s: self.orginal_img_size,:] = depth_org_img[:, 0: self.orginal_img_size - p_s,:]
                depth_aug_img[:, 0: p_s,:] = depth_org_img[:, self.orginal_img_size - p_s: self.orginal_img_size,:]
        else:
            # if device_number == 'device_1':
            #     axis = 'x'
            # else:
            #     axis = 'z'
            axis = 'x'
            r_s = np.random.randint(1, 3)
            if r_s == 1:
                # left side aumentation
                try:
                    if bbox[0] > max_r:
                        p_s = np.random.randint(1, max_r)
                    else:
                        p_s = np.random.randint(1, bbox[0])
                except:
                    p_s = 1
                aug_pixel = (-1 * p_s)/(self.orginal_img_size/self.final_img_size)
                depth_aug_img[:, :, 0: self.orginal_img_size - p_s] = depth_org_img[:, :, p_s:self.orginal_img_size]
                depth_aug_img[:, :, self.orginal_img_size - p_s: self.orginal_img_size] = depth_org_img[:, :, 0:p_s]
            else:
                # right side aumentation
                try:
                    if self.orginal_img_size - (bbox[0] + w) > max_r:
                        p_s = np.random.randint(1, max_r)
                    else:
                        p_s = np.random.randint(1, self.orginal_img_size - (bbox[0] + w))
                except:
                    p_s = 1
                aug_pixel = p_s/(self.orginal_img_size/self.final_img_size)
                depth_aug_img[:, :, p_s: self.orginal_img_size] = depth_org_img[:, :, 0: self.orginal_img_size - p_s]
                depth_aug_img[:, :, 0: p_s] = depth_org_img[:, :, self.orginal_img_size - p_s: self.orginal_img_size]

        return depth_aug_img, axis, aug_pixel

    def load_depth_data(self, dense_path, depth_path, img_name):

        a_f = open(dense_path + '/' + img_name + '.pkl', 'rb')
        a_data = pickle.load(a_f)
        cmap = plt.get_cmap('jet')
        filter = ImageFilter.MaxFilter(size=3)

        for i in range(len(a_data)-1):
            a_bbox = a_data[i][0].int()
            a_ch, a_h, a_w = a_data[i][1].size()
            if i >= 2:
                a_bbox[0] = torch.min(a_bbox[0], a_data[i-1][0].int())
                a_bbox[1] = torch.min(a_bbox[1], a_data[i-1][1].int())

                a_w = torch.max(a_bbox[0], a_data[i-1][0].int()) - a_bbox[0] + 1
                a_h = torch.max(a_bbox[1], a_data[i-1][1].int()) - a_bbox[1] + 1
        depth_img_name = 'MDepth-' + str(int(img_name[5:11])).zfill(8) + '.png'
        try:
            depth_img = Image.open(os.path.join(depth_path,  depth_img_name)).filter(self.filter).resize((self.orginal_img_size, self.orginal_img_size), Image.ANTIALIAS)
        except:
            new_img_name = 'MDepth-' + str(int(depth_img_name[6:15]) + 1).zfill(8) + '.png'
            if os.path.exists(os.path.join(depth_path,  new_img_name)):
                depth_img = Image.open(os.path.join(depth_path,  new_img_name)).filter(self.filter).resize((self.orginal_img_size, self.orginal_img_size), Image.ANTIALIAS)
            else:
                new_img_name = 'MDepth-' + str(int(depth_img_name[6:15]) - 1).zfill(8) + '.png'
                depth_img = Image.open(os.path.join(depth_path,  new_img_name)).filter(self.filter).resize((self.orginal_img_size, self.orginal_img_size), Image.ANTIALIAS)

        np_depth_img = np.array(depth_img)
        mx = np.max(np_depth_img)
        np_depth_img2 = np_depth_img/mx
        f_depth_img = cmap(np_depth_img2)[..., :3]


        return self.transform(f_depth_img), a_bbox, a_w, a_h

    def load_depth_data2(self, dense_path, depth_path, img_name):

        # filter = ImageFilter.MaxFilter(size=3)
        a_f = open(dense_path + '/' + img_name + '.pkl', 'rb')
        a_data = pickle.load(a_f)
        cmap = plt.get_cmap('jet')
        for i in range(len(a_data)-1):
            a_bbox = a_data[i][0].int()
            a_ch, a_h, a_w = a_data[i][1].size()

        depth_img_name = 'MDepth-' + str(int(img_name[5:11])).zfill(8) + '.png'
        try:
            depth_img = Image.open(os.path.join(depth_path,  depth_img_name)).resize((self.orginal_img_size, self.orginal_img_size), Image.ANTIALIAS)
        except:
            new_img_name = 'MDepth-' + str(int(img_name[5:11]) + 1).zfill(8) + '.png'
            if os.path.exists(os.path.join(depth_path,  new_img_name)):
                depth_img = Image.open(os.path.join(depth_path,  new_img_name)).resize((self.orginal_img_size, self.orginal_img_size), Image.ANTIALIAS)
            else:
                new_img_name = 'MDepth-' + str(int(img_name[5:11]) - 1).zfill(8) + '.png'
                depth_img = Image.open(os.path.join(depth_path,  new_img_name)).resize((self.orginal_img_size, self.orginal_img_size), Image.ANTIALIAS)
        np_depth_img = np.array(depth_img)
        mx = np.max(np_depth_img)
        np_depth_img = np_depth_img/mx
        f_depth_img = cmap(np_depth_img)[..., :3]

        return self.transform(f_depth_img), a_bbox, a_w, a_h

    def random_distance_frame(self, idx, ra_f_idx, a_person_name, a_rr_action_name):
        a_sheet = self.file.iloc
        flag = True
        i = 1
        while flag:
            if idx + ra_f_idx < len(self.file):
                a2_person_name = a_sheet[idx + ra_f_idx, 2]
                a2_rr_action_name = a_sheet[idx + ra_f_idx, 3]
                a2_img_name = a_sheet[idx + ra_f_idx, 6]
                if a2_person_name != a_person_name or a2_rr_action_name != a_rr_action_name:
                    ra_f_idx = -1 * ra_f_idx
                    if i % 2 == 0:
                        ra_f_idx = ra_f_idx//2
                else:
                    flag = False
                i = i + 1
            else:
                ra_f_idx = len(self.file) - idx - 1

        return ra_f_idx, a2_img_name, a2_person_name, a2_rr_action_name

    def change_view(self, input):
        return input.view(-1, self.final_img_size, self.final_img_size)







