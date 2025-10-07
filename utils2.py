import numpy as np
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", 'tiff', 'bmp', '.mat'])


def load_img(filepath):
    img = Image.open(filepath)  # .convert('RGB')
    return img


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))


def save_np_img(image_numpy, filename):
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))


def record_loss(loss_csv, epoch, loss0=0, loss1=0, loss2=0, loss3=0, loss4=0, loss5=0, loss6=0, loss7=0, loss8=0):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{},{},{},{},{}\n'.format(epoch, loss0, loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8))

    loss_csv.flush()
    loss_csv.close


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = gray.unsqueeze(2)
    return gray


def rgb2gray_tensor(rgb):
    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = gray.unsqueeze(1)
    return gray


def save_gray_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = image_numpy * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))
