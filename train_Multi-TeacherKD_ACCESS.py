from __future__ import print_function
import os
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils2 import *
from networks_dense import UNetKDPhase, UNetKD, UNetKDPhaseParallel
from dataset import *
from loss import vgg16_loss
from loss.fd_loss import OFD, OFD2

# Training settings
parser = argparse.ArgumentParser(description='Multi-TeacherFD for colorization')
parser.add_argument('--batch_size', type=int, default=5, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=True, help='use cuda?')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
parser.add_argument('--teacher', type=bool, default=True, help='Use the pre-trained teacher')
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

# channel 수 수정
detail_teacher = UNetKDPhase(in_ch=4, out_ch=1, bilinear=False).cuda()
detail_teacher = torch.nn.DataParallel(detail_teacher)
color_teacher = UNetKD(in_ch=6, out_ch=3, bilinear=False).cuda()
color_teacher = torch.nn.DataParallel(color_teacher)

if opt.teacher:
    # Detail-Teacher
    checkpoint1 = torch.load('/net_epoch_100.pth')
    detail_teacher.load_state_dict(checkpoint1['model_state_dict'])
    print('Use the pretrained detail-teacher model!')

    # Color-Teacher
    checkpoint2 = torch.load('/net_epoch_100.pth')
    color_teacher.load_state_dict(checkpoint2['model_state_dict'])
    print('Use the pretrained color-teacher Model!')

student = UNetKDPhaseParallel(in_ch=3, out_ch=3, bilinear=False).cuda()
student = torch.nn.DataParallel(student)

L1_loss = torch.nn.L1Loss().to(device)
VGG_loss = vgg16_loss.VGG16Loss().to(device)
FD_loss6 = OFD(in_channels=512, out_channels=512).to(device)
FD_loss7 = OFD(in_channels=256, out_channels=256).to(device)
FD_loss8 = OFD(in_channels=128, out_channels=128).to(device)
FD_loss9 = OFD(in_channels=64, out_channels=64).to(device)

FD_loss6_2 = OFD2(in_channels=512, out_channels=512).to(device)
FD_loss7_2 = OFD2(in_channels=256, out_channels=256).to(device)
FD_loss8_2 = OFD2(in_channels=128, out_channels=128).to(device)
FD_loss9_2 = OFD2(in_channels=64, out_channels=64).to(device)

optimizer = optim.Adam(student.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.1)

SAVE_FOLDER = ''    # Path for saving model

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

net_path = SAVE_FOLDER + "/net_epoch_{}.pth".format(opt.epoch_count - 1)
loss_csv = open(SAVE_FOLDER + '/loss.csv', 'w+')
loss_csv.write('{},{}\n'.format('epoch', 'loss'))

if os.path.isfile(net_path):
    checkpoint = torch.load(net_path)
    student.load_state_dict(checkpoint['model_state_dict'])
    print('successfully loaded checkpoints!')

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    # train
    for param_group in optimizer.param_groups:
        print('학습률: ', param_group['lr'])

    loss_epoch = 0

    # data: input image, target: target image
    for iteration, (data, target) in enumerate(dataloader):
        # forward
        data = data.cuda()
        data = data[:, :, :, :]
        target = target.cuda()

        # for gray gt
        gray_target = rgb2gray_tensor(target)

        optimizer.zero_grad()

        for p1 in detail_teacher.parameters():
            p1.requires_grad = False

        for p2 in color_teacher.parameters():
            p2.requires_grad = False

        pha6_t, pha7_t, pha8_t, pha9_t, _ = detail_teacher(torch.cat([data, gray_target], dim=1))
        _, _, _, _, _, x6_t, x7_t, x8_t, x9_t, _ = color_teacher(torch.cat([data, target], dim=1))
        pha6, pha7, pha8, pha9, x6_2, x7_2, x8_2, x9_2, out = student(data)

        l1_loss = L1_loss(out, target)
        vgg_loss = VGG_loss(out, target)
        distil_loss = FD_loss6(x6_2, x6_t) + FD_loss7(x7_2, x7_t) + FD_loss8(x8_2, x8_t) + FD_loss9(x9_2, x9_t)  # color distil
        distil_loss += FD_loss6_2(pha6, pha6_t) + FD_loss7_2(pha7, pha7_t) + FD_loss8_2(pha8, pha8_t) + FD_loss9_2(pha9, pha9_t)  # detail distil

        loss = l1_loss + vgg_loss + 0.001 * distil_loss

        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()

        print("===> Epoch[{}]({}/{}): vgg_loss: {:.7f} | l1_loss: {:.7f} | distil_loss: {:.7f} | loss: {:.7f}".format(epoch, iteration, len(dataloader), vgg_loss.item(), l1_loss.item(), distil_loss.item(), loss.item()))

    # scheduler.step()
    loss_epoch = loss_epoch / (len(dataloader))
    print("===> Epoch[{}]: loss: {:.7f}".format(epoch, loss_epoch))

    record_loss(loss_csv, epoch, loss_epoch)

    # checkpoint
    if epoch % 100 == 0:
        net_path = SAVE_FOLDER + "/net_epoch_{}.pth".format(epoch)
        torch.save({'epoch': epoch,
                    'model_state_dict': student.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, net_path)
        print("Checkpoint saved to {}".format(net_path))
