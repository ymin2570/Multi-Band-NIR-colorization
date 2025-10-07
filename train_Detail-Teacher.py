import argparse

import torch.backends.cudnn as cudnn
from dataset import *
from torch.utils.data import DataLoader
from networks_dense import UNetKDPhase
import torch.optim as optim
from utils2 import *
from loss import vgg16_loss
import os

# Training settings
parser = argparse.ArgumentParser(description='detail-teacher for colorization')
parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=True, help='use cuda?')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
parser.add_argument('--gray_pretrained', type=bool, default=False, help='Use the Gray-pretrained Model')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)

if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_data = DatasetFromFolder()

print('train data num: ', len(train_data))

dataloader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, pin_memory=True, drop_last=True)

device = "cuda"

net = UNetKDPhase(in_ch=4, out_ch=1, bilinear=False).cuda()
net = torch.nn.DataParallel(net)

if opt.gray_pretrained:
    checkpoint = torch.load('')
    net.load_state_dict(checkpoint['model_state_dict'])
    print('Use the Gray-pretrained Model!')

L1_loss = torch.nn.L1Loss().to(device)
VGG_loss = vgg16_loss.VGG16Loss().to(device)

optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

SAVE_FOLDER = ''    # Path for saving model

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

net_path = SAVE_FOLDER + "/net_epoch_{}.pth".format(opt.epoch_count - 1)
loss_csv = open(SAVE_FOLDER + '/loss.csv', 'w+')
loss_csv.write('{},{}\n'.format('epoch', 'loss'))

if os.path.isfile(net_path):
    checkpoint = torch.load(net_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    print('successfully loaded checkpoints!')

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    # train
    for param_group in optimizer.param_groups:
        print('learning rate: ', param_group['lr'])

    loss_epoch = 0

    for iteration, (data, target) in enumerate(dataloader):
        # forward
        data = data.cuda()
        data = data[:, :, :, :]
        target = target.cuda()

        # for gray image
        gray_target = rgb2gray_tensor(target)

        """ NETWORK """
        optimizer.zero_grad()
        _, _, _, _, out = net(torch.cat([data, gray_target], dim=1))

        l1_loss = L1_loss(out, gray_target)
        vgg_loss = VGG_loss(out, gray_target)

        loss = l1_loss + vgg_loss

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        print("===> Epoch[{}]({}/{}): vgg_loss: {:.7f} | l1_loss: {:.7f} | loss: {:.7f}".format(epoch, iteration, len(dataloader), vgg_loss.item(), l1_loss.item(), loss.item()))

    loss_epoch = loss_epoch / (len(dataloader))

    record_loss(loss_csv, epoch, loss_epoch)

    # save checkpoint
    if epoch % 100 == 0:
        net_path = SAVE_FOLDER + "/net_epoch_{}.pth".format(epoch)
        torch.save({'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, net_path)
        print('Checkpoint saved to {}"'.format(net_path))
