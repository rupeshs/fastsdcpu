import cv2
import numpy as np

from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


def get_scale_from_dataset(dataset: Dataset):
    lr, hr = dataset[0]
    dim1 = round(hr.shape[1] / lr.shape[1])
    dim2 = round(hr.shape[2] / lr.shape[2])
    scale = max(dim1, dim2)
    return scale


def convert_rgb_to_y(img, dim_order='hwc'):
    if dim_order == 'hwc':
        return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    else:
        return 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.


def denormalize(img):
    img = img.mul(255.0).clamp(0.0, 255.0)
    return img


def calc_psnr(img1, img2, max=255.0):
    return 10. * ((max ** 2) / ((img1 - img2) ** 2).mean()).log10()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_ssim(img1, img2):
    """calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    """
    img1=img1.cpu().numpy()
    img2 = img2.cpu().numpy()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def compute_metrics(eval_prediction, scale):
    preds = eval_prediction.predictions
    labels = eval_prediction.labels

    # from piq import ssim, psnr
    # print(psnr(denormalize(preds), denormalize(labels), data_range=255.),
    #       ssim(denormalize(preds), denormalize(labels), data_range=255.))

    # original = preds[0][0][0][0]

    preds = convert_rgb_to_y(denormalize(preds.squeeze(0)), dim_order='chw')
    labels = convert_rgb_to_y(denormalize(labels.squeeze(0)), dim_order='chw')

    # print(preds[0][0], original * 255.)

    preds = preds[scale:-scale, scale:-scale]
    labels = labels[scale:-scale, scale:-scale]

    # print(calc_psnr(preds, labels), calc_ssim(preds, labels))

    return {
        'psnr': calc_psnr(preds, labels),
        'ssim': calc_ssim(preds, labels)
    }


def calculate_mean_std(dataset):
    image_loader = DataLoader(dataset,
                              batch_size=1,
                              shuffle=False,
                              pin_memory=True)

    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])
    count = 0

    for inputs in tqdm(image_loader, desc='Calculating RGB pixel mean and std'):
        lr, hr = inputs
        psum += hr.sum(axis=[0, 2, 3])
        psum_sq += (hr ** 2).sum(axis=[0, 2, 3])
        count += hr.shape[2] * hr.shape[3]

    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    np.set_printoptions(precision=4)
    print(f'mean: {str(total_mean.numpy())}')
    print(f'std: {str(total_std.numpy())}')
