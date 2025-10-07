import easydict
from networks_dense import *
import os
import scipy.io
from utils2 import *

def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repeat = 28
timings = np.zeros((repeat,1))

for TEST_EPOCH in range(100, 100 + 100, 100):
    MODEL_PATH = "/net_epoch_{}.pth".format(TEST_EPOCH)  # model path
    IMAGE_DIR = 'test/nir_3ch_mat/'  # NIR image

    # Testing settings
    opt = easydict.EasyDict({
        "dataset": 'epoch_{}'.format(TEST_EPOCH),
        "cuda": True,
    })
    print(opt)

    device = torch.device("cuda:0" if opt.cuda else "cpu")
    print(device)

    net = UNetKDPhaseParallel(in_ch=3, out_ch=3, bilinear=False).cuda()
    net = torch.nn.DataParallel(net)

    if os.path.isfile(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH)
        net.load_state_dict(checkpoint['model_state_dict'])
        print('successfully loaded checkpoints!')

    image_filenames = [x for x in os.listdir(IMAGE_DIR) if is_mat_file(x)]

    net.eval()
    with torch.no_grad():
        for image_name in image_filenames:
            mat = scipy.io.loadmat(IMAGE_DIR + image_name)
            mat = mat['mat_data']
            # mat = np.expand_dims(mat, axis=2)  # for NIR 1band
            mat = torch.from_numpy(mat)
            data = mat.float()  # change type to your use case
            data = data / 255.
            data.transpose_(0, 2).transpose_(1, 2).unsqueeze_(0)
            # breakpoint()
            data = data[:, :, :, :]

            """ NETWORK """
            x6, x7, x8, x9, x6_2, x7_2, x8_2, x9_2, out = net(data)

            out = out.detach().squeeze(0).cpu()

            SAVE_PATH = os.path.join("", opt.dataset)    # Path for saving results

            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)

            save_img(out, "{}/{}".format(SAVE_PATH, image_name[:-4] + '.bmp'))
