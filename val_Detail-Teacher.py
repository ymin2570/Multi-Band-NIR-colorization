from __future__ import print_function
import os
import torch
from utils2 import save_img, save_gray_img, rgb2gray_tensor
import easydict
from networks_dense import UNet
from dataset import DatasetFromFolder_test
from torch.utils.data import DataLoader


def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])


print('===> Loading datasets')
test_data = DatasetFromFolder_test()
print('test data num: ', len(test_data))
dataloader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, pin_memory=True, drop_last=True)


TEST_EPOCH = 100
MODEL_PATH = "/net_epoch_{}.pth".format(TEST_EPOCH)  # model path

# Testing settings
opt = easydict.EasyDict({
    "dataset": 'epoch_{}'.format(TEST_EPOCH),
    "cuda": True,
})
print(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")
print(device)

net = UNet(in_ch=4, out_ch=1, bilinear=False).cuda()
net = torch.nn.DataParallel(net)

if os.path.isfile(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    print('successfully loaded checkpoints!')


net.eval()
with torch.no_grad():
    for iteration, (data, target) in enumerate(dataloader):
        # forward
        data = data.cuda()
        data = data[:, :, :, :]
        target = target.cuda()

        # for gray image
        gray_target = rgb2gray_tensor(target)

        """ NETWORK """
        out = net(torch.cat([data, gray_target], dim=1))

        out = out.detach().squeeze(0).squeeze(0).cpu()

        SAVE_PATH = os.path.join("", opt.dataset)    # Path for saving results

        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)

        save_gray_img(out, "{}/{}".format(SAVE_PATH, str(iteration + 1).zfill(4) + '.bmp'))
