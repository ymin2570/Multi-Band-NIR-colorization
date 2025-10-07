import torch
from os.path import join
import torch.utils.data as data
from utils2 import load_img
import numpy as np
import scipy
from scipy import io
from os import listdir


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])


class DatasetFromFolder(data.Dataset):  # 3ch NIR, 256x256 patch
    def __init__(self):
        super(DatasetFromFolder, self).__init__()
        self.label_dir = 'train/gt_rgb or gt_gray/'      # GT image (RGB or GRAY)
        self.data_dir = 'train/nir_3ch_mat/'  # NIR 3ch image
        self.data_list = [join(self.data_dir, x) for x in listdir(self.data_dir) if is_mat_file(x)]
        self.label_list = listdir(self.label_dir)

    def __getitem__(self, index):
        img = load_img(self.label_dir + self.label_list[index])
        img = np.asarray(img)
        h, w, _ = img.shape
        h_ = np.random.randint(0, h - 256)
        w_ = np.random.randint(0, w - 256)

        label = torch.from_numpy(img)
        label = label[h_:h_ + 256, w_:w_ + 256, :]

        mat = scipy.io.loadmat(self.data_list[index])
        mat = mat['mat_data']
        data = torch.from_numpy(mat)

        data = data[h_:h_ + 256, w_:w_ + 256, :]
        data = data.float()  # change type to your use case
        data = data / 255.

        label = label / 255.
        data.transpose_(0, 2).transpose_(1, 2)
        label.transpose_(0, 2).transpose_(1, 2)

        return data.float(), label.float()

    def __len__(self):
        return len(self.label_list)


class DatasetFromFolder2(data.Dataset):  # 1ch NIR, 256x256 patch
    def __init__(self):
        super(DatasetFromFolder2, self).__init__()
        self.label_dir = 'train/gt_rgb or gt_gray/'  # GT image (RGB or GRAY)
        self.data_dir = 'train/nir_1ch/'  # NIR 1ch image
        self.data_list = listdir(self.data_dir)
        self.label_list = listdir(self.label_dir)

    def __getitem__(self, index):
        img = load_img(self.label_dir + self.label_list[index])
        img = np.asarray(img)

        h, w, _ = img.shape

        if h <= 256:
            h_ = 0
        else:
            h_ = np.random.randint(0, h - 256)

        if w <= 256:
            w_ = 0
        else:
            w_ = np.random.randint(0, w - 256)

        label = torch.from_numpy(img)
        label = label[h_:h_ + 256, w_:w_ + 256, :]

        data = load_img(self.data_dir + self.data_list[index])
        data = np.asarray(data)

        data = data[:, :, 1]
        data = np.expand_dims(data, axis=2)
        data = data[h_:h_ + 256, w_:w_ + 256, :]

        data = torch.from_numpy(data)

        data = data.float()  # change type to your use case
        data = data / 255.
        label = label / 255.

        data.transpose_(0, 2).transpose_(1, 2)
        label.transpose_(0, 2).transpose_(1, 2)
        return data.float(), label.float()

    def __len__(self):
        return len(self.label_list)


class DatasetFromFolder_test(data.Dataset):  # 3ch NIR for test
    def __init__(self):
        super(DatasetFromFolder_test, self).__init__()
        self.label_dir = 'test/gt_rgb or gt_gray/'      # GT image (RGB or GRAY)
        self.data_dir = 'test/nir_3ch_mat/'  # NIR 3ch image
        self.data_list = [join(self.data_dir, x) for x in listdir(self.data_dir) if is_mat_file(x)]
        self.label_list = listdir(self.label_dir)

    def __getitem__(self, index):
        img = load_img(self.label_dir + self.label_list[index])
        img = np.asarray(img)
        label = torch.from_numpy(img)

        mat = scipy.io.loadmat(self.data_list[index])
        mat = mat['mat_data']
        data = torch.from_numpy(mat)
        data = data.float()  # change type to your use case
        data = data / 255.

        label = label / 255.
        data.transpose_(0, 2).transpose_(1, 2)
        label.transpose_(0, 2).transpose_(1, 2)

        return data.float(), label.float()

    def __len__(self):
        return len(self.label_list)


class DatasetFromFolder_test2(data.Dataset):  # 1ch NIR for test
    def __init__(self):
        super(DatasetFromFolder_test2, self).__init__()
        self.label_dir = 'test/gt_rgb or gt_gray/'  # GT image (RGB or GRAY)
        self.data_dir = 'test/nir_1ch/'  # NIR 1ch image
        self.data_list = listdir(self.data_dir)
        self.label_list = listdir(self.label_dir)

    def __getitem__(self, index):
        img = load_img(self.label_dir + self.label_list[index])
        img = np.asarray(img)

        label = torch.from_numpy(img)

        data = load_img(self.data_dir + self.data_list[index])
        data = np.asarray(data)

        data = data[:, :, 1]
        data = np.expand_dims(data, axis=2)

        data = torch.from_numpy(data)

        data = data.float()  # change type to your use case
        data = data / 255.
        label = label / 255.

        data.transpose_(0, 2).transpose_(1, 2)
        label.transpose_(0, 2).transpose_(1, 2)
        return data.float(), label.float()

    def __len__(self):
        return len(self.label_list)