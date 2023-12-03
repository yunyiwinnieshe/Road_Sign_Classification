import os
import random
import pandas as pd
import numpy as np

import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import optuna


class Setup:
    def __init__(self, image_path, anno_path):
        self.image_path = image_path
        self.anno_path = anno_path
        self.annotations = [os.path.join(dir_path, file) for dir_path, dir_name, files in os.walk(anno_path)
                            for file in files if file.endswith('.xml')]
        self.train_df = None

    def generate_train_df(self):
        anno_list = []

        for path in self.annotations:
            root = ET.parse(path).getroot()
            anno = {}
            anno['filename'] = Path(str(self.image_path) + '/' + root.find('./filename').text)
            anno['width'] = root.find('./size/width').text
            anno['height'] = root.find('./size/height').text
            anno['class'] = root.find('./object/name').text
            anno['xmin'] = int(root.find('./object/bndbox/xmin').text)
            anno['ymin'] = int(root.find('./object/bndbox/ymin').text)
            anno['xmax'] = int(root.find('./object/bndbox/xmax').text)
            anno['ymax'] = int(root.find('./object/bndbox/ymax').text)
            anno_list.append(anno)

        df = pd.DataFrame(anno_list)
        class_dict = {'speedlimit': 0, 'stop': 1, 'crosswalk': 2, 'trafficlight': 3}
        df['class'] = df['class'].apply(lambda x: class_dict[x])
        self.train_df = df

        self.__add_new_paths(300)
        return self.train_df

    def test_mask(self, row):
        im = cv2.imread(str(self.train_df.values[row][0]))
        bb = self.__create_bb_array(self.train_df.values[row])
        print(im.shape)

        Y = self.__create_mask(bb, im)
        # self.__mask_to_bb(Y)
        return im, Y

    def test_bb(self, row):
        im = cv2.imread(str(self.train_df['new_path'].values[row]))
        print(str(self.train_df.values[row][8]))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        self.__show_corner_bb(im, self.train_df['new_bb'].values[row])

    def test_transforms(self, row):
        im, bb = self.__transformsXY(str(self.train_df['new_path'].values[row]),
                                       self.train_df['new_bb'].values[row],
                                       is_transforms=True)
        self.__show_corner_bb(im, bb)

    def transforms(self, path, bb, is_transforms):
        return self.__transformsXY(path, bb, is_transforms)

    def __read_image(self, path):
        return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

    def __create_mask(self, bb, image):
        rows, cols, *_ = image.shape
        Y = np.zeros((rows, cols))
        bb = bb.astype(np.int)
        Y[bb[0]:bb[2], bb[1]:bb[3]] = 1.
        return Y

    def __mask_to_bb(self, Y):
        cols, rows = np.nonzero(Y)

        if len(cols) == 0:
            return np.zeros(4, dtype=np.float32)

        top_row = np.min(rows)
        bottom_row = np.max(rows)
        left_col = np.min(cols)
        right_col = np.max(cols)

        return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)

    def __create_bb_array(self, x):
        return np.array([x[5], x[4], x[7], x[6]])

    def __resize_image_bb(self, read_path, write_path, bb, size):
        im = self.__read_image(read_path)
        im_resized = cv2.resize(im, (size, size))
        Y_resized = cv2.resize(self.__create_mask(bb, im), (size, size))
        new_path = str(write_path/read_path.parts[-1])
        cv2.imwrite(new_path, cv2.cvtColor(im_resized, cv2.COLOR_RGB2BGR))
        return new_path, self.__mask_to_bb(Y_resized)

    def __add_new_paths(self, im_size):
        new_paths = []
        new_bbs = []
        train_path_resized = Path('./images_resized')
        Path.mkdir(train_path_resized, exist_ok=True)

        for index, row in self.train_df.iterrows():
            new_path, new_bb = self.__resize_image_bb(row['filename'],
                                                      train_path_resized,
                                                      self.__create_bb_array(row.values),
                                                      im_size)
            new_paths.append(new_path)
            new_bbs.append(new_bb)

        self.train_df['new_path'] = new_paths
        self.train_df['new_bb'] = new_bbs

    def __crop(self, im, row, col, targ_row, targ_col):
        return im[row:row + targ_row, col:col + targ_col]

    def __center_crop(self, im, r_pix=8):
        r, c, *_ = im.shape
        c_pix = round(r_pix * c / r)
        return self.__crop(im, r_pix, c_pix, r - 2*r_pix, c - 2*c_pix)

    def __rotate_cv(self, im, deg, y=False, mode=cv2.BORDER_REFLECT):
        r, c, *_ = im.shape
        M = cv2.getRotationMatrix2D((c/2, r/2), deg, 1)
        if y:
            return cv2.warpAffine(im, M, (c, r), borderMode=cv2.BORDER_CONSTANT)
        return cv2.warpAffine(im, M, (c, r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS)

    def __random_cropXY(self, x, Y, r_pix=8):
        r, c, *_ = x.shape
        c_pix = round(r_pix * c/r)
        rand_r = random.uniform(0, 1)
        rand_c = random.uniform(0, 1)
        start_r = np.floor(2 * rand_r * r_pix).astype(int)
        start_c = np.floor(2 * rand_c * c_pix).astype(int)
        xx = self.__crop(x, start_r, start_c, r - 2*r_pix, c - 2*c_pix)
        YY = self.__crop(Y, start_r, start_c, r - 2*r_pix, c - 2*c_pix)
        return xx, YY

    def __transformsXY(self, path, bb, is_transforms):
        x = cv2.imread(str(path)).astype(np.float32)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB) / 255
        Y = self.__create_mask(bb, x)
        if is_transforms:
            rdeg = (np.random.random() - 0.50) * 20
            x = self.__rotate_cv(x, rdeg)
            Y = self.__rotate_cv(Y, rdeg, y=True)
            if np.random.random() > 0.5:
                x = np.fliplr(x).copy()
                Y = np.fliplr(Y).copy()

            x, Y = self.__random_cropXY(x, Y)
        else:
            x, Y = self.__center_crop(x), self.__center_crop(Y)
        return x, self.__mask_to_bb(Y)

    def __create_corner_rect(self, bb, color='red'):
        bb = np.array(bb, dtype=np.float32)
        return plt.Rectangle((bb[1], bb[0]), bb[3] - bb[1], bb[2] - bb[0], color=color, fill=False, lw=3)

    def __show_corner_bb(self, im, bb):
        plt.imshow(im)
        plt.gca().add_patch(self.__create_corner_rect(bb))


class RoadData(Dataset):
    def __init__(self, paths, bb, y, s: Setup, is_transforms=False):
        self.paths = paths.values
        self.y = y.values
        self.bb = bb.values
        self.is_transforms = is_transforms
        self.setup = s

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, key):
        path = self.paths[key]
        y_class = self.y[key]
        x, y_bb = self.setup.transforms(path, self.bb[key], self.is_transforms)
        x = self.__normalize(x)
        x = np.rollaxis(x, 2)
        return x, y_class, y_bb

    def __normalize(self, image):
        stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
        return (image - stats[0]) / stats[1]


class RoadModel(nn.Module):
    def __init__(self, trial, num_conv_layers, num_filters, num_neurons, drop_conv2, drop_fc1):
        super(RoadModel, self).__init__()
        in_size = 300       # image input size (300 pixels)
        kernel_size = 5     # convolution filter size

        self.convs = nn.ModuleList([nn.Conv2d(3, num_filters[0], kernel_size=(5, 5))])
        out_size = in_size - kernel_size + 1    # output kernel size
        out_size = int(out_size / 2)    # after pooling

        for i in range(1, num_conv_layers):
            self.convs.append(nn.Conv2d(in_channels=num_filters[i-1], out_channels=num_filters[i], kernel_size=(5, 5)))
            out_size = out_size - kernel_size + 1
            out_size = int(out_size / 2)

        self.conv2_drop = nn.Dropout2d(p=drop_conv2)        # dropout for conv2
        self.out_feature = num_filters[num_conv_layers-1] * out_size * out_size     # size of flattened features
        self.fc1 = nn.Linear(self.out_feature, num_neurons)     # fully connected layer 1
        self.fc2 = nn.Linear(num_neurons, 4)       # fully connected layer 2
        self.fc3 = nn.Linear(num_neurons, 4)
        self.p1 = drop_fc1      # dropout ratio for FC1

        for i in range(1, num_conv_layers):
            nn.init.kaiming_normal_(self.convs[i].weight, nonlinearity='relu')
            if self.convs[i].bias is not None:
                nn.init.constant_(self.convs[i].bias, 0)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')

    def forward(self, x):
        for i, conv_i in enumerate(self.convs):     # for each convolutional layer
            if i == 2:      # add dropout if layer 2
                x = F.relu(F.max_pool2d(self.conv2_drop(conv_i(x)), 2))     # conv_i, dropout, max-pooling, relu
            else:
                x = F.relu(F.max_pool2d(conv_i(x), 2))      # conv_i, max-pooling, relu

        x = x.view(-1, self.out_feature)        # flatten tensor
        x = F.relu(self.fc1(x))         # fc1, relu
        x = F.dropout(x, p=self.p1, training=self.training)     # apply dropout after fc1 only when training
        bb = self.fc3(x)        # for the bounding box
        x = self.fc2(x)         # for the classifier

        return F.log_softmax(x, dim=1), bb    # log(softmax(x))


if __name__ == "__main__":
    setup = Setup('../images', '../annotations')
    train_df = setup.generate_train_df()
    print(train_df['class'].value_counts())
    print(train_df.shape)

    print(train_df.head())
    setup.test_mask(58)