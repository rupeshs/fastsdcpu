import h5py
import random
import numpy as np
from PIL import Image
from pathlib import Path

from torch.utils.data import Dataset
from torchvision.transforms import transforms


DIV2K_RGB_MEAN = (0.4488, 0.4371, 0.4040)
DIV2K_RGB_STD = (1.0, 1.0, 1.0)


def augment_five_crop(batch, scale=None):
    hr_augment_path = None
    lr_augment_path = None
    if scale is None:
        if len(batch['hr']) > 0:
            scale = get_scale(Image.open(batch['lr'][0]), Image.open(batch['hr'][0]))
            hr_augment_path = Path(batch['hr'][0]).parent / f'augment_x{scale}'
            lr_augment_path = Path(batch['lr'][0]).parent / f'augment_x{scale}'
            hr_augment_path.mkdir(parents=True, exist_ok=True)
            lr_augment_path.mkdir(parents=True, exist_ok=True)
    outputs_lr = []
    outputs_hr = []
    for idx, example in enumerate(batch['hr']):
        hr_path = Path(example)
        lr_path = Path(batch['lr'][idx])
        hr = Image.open(hr_path).convert('RGB')
        for aug_idx, hr in enumerate(transforms.FiveCrop(size=(hr.height // 2, hr.width // 2))(hr)):
            hr = hr.resize(((hr.width // scale) * scale, (hr.height // scale) * scale),
                           resample=Image.BICUBIC)
            lr = hr.resize((hr.width // scale, hr.height // scale), resample=Image.BICUBIC)
            hr_file_path = hr_augment_path / f'{hr_path.stem}_{aug_idx}{hr_path.suffix}'
            lr_file_path = lr_augment_path / f'{lr_path.stem}_{aug_idx}{lr_path.suffix}'
            hr.save(hr_file_path, 'PNG')
            lr.save(lr_file_path, 'PNG')
            outputs_hr.append(hr_file_path.as_posix())
            outputs_lr.append(lr_file_path.as_posix())
    return {
        'lr': outputs_lr,
        'hr': outputs_hr
    }


def get_scale_from_dataset(dataset):
    scale = None
    if len(dataset) > 0:
        lr = Image.open(dataset[0]['lr'])
        hr = Image.open(dataset[0]['hr'])
        dim1 = round(hr.width / lr.width)
        dim2 = round(hr.height / lr.height)
        scale = max(dim1, dim2)
    return scale


def get_scale(lr, hr):
    dim1 = round(hr.width / lr.width)
    dim2 = round(hr.height / lr.height)
    scale = max(dim1, dim2)
    return scale


def resize_image(lr_image, hr_image, scale=None):
    if scale is None:
        scale = get_scale(lr_image, hr_image)
    if lr_image.width * scale != hr_image.width or lr_image.height * scale != hr_image.height:
        hr_width = lr_image.width * scale
        hr_height = lr_image.height * scale
        return hr_image.resize((hr_width, hr_height), resample=Image.BICUBIC)
    return hr_image


class EvalDataset(Dataset):
    def __init__(self, dataset):
        super(EvalDataset, self).__init__()
        self.dataset = dataset
        self.scale = get_scale_from_dataset(dataset)

    def __getitem__(self, idx):
        lr_image = Image.open(self.dataset[idx]['lr']).convert('RGB')
        hr_image = resize_image(lr_image, Image.open(self.dataset[idx]['hr']).convert('RGB'), scale=self.scale)
        lr = np.array(lr_image)
        hr = np.array(hr_image)
        lr = lr.astype(np.float32).transpose([2, 0, 1]) / 255.0
        hr = hr.astype(np.float32).transpose([2, 0, 1]) / 255.0
        return lr, hr

    def __len__(self):
        return len(self.dataset)


class EvalDatasetH5(Dataset):
    def __init__(self, h5_file):
        super(EvalDatasetH5, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr = f['lr'][str(idx)][::].astype(np.float32).transpose([2, 0, 1]) / 255.0
            hr = f['hr'][str(idx)][::].astype(np.float32).transpose([2, 0, 1]) / 255.0
            return lr, hr

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class TrainDataset(Dataset):
    def __init__(self, dataset, patch_size=64):
        super(TrainDataset, self).__init__()
        self.dataset = dataset
        self.patch_size = patch_size
        self.scale = get_scale_from_dataset(dataset)

    @staticmethod
    def random_crop(lr, hr, size, scale):
        lr_left = random.randint(0, lr.shape[1] - size)
        lr_right = lr_left + size
        lr_top = random.randint(0, lr.shape[0] - size)
        lr_bottom = lr_top + size
        hr_left = lr_left * scale
        hr_right = lr_right * scale
        hr_top = lr_top * scale
        hr_bottom = lr_bottom * scale
        lr = lr[lr_top:lr_bottom, lr_left:lr_right]
        hr = hr[hr_top:hr_bottom, hr_left:hr_right]
        return lr, hr

    @staticmethod
    def random_horizontal_flip(lr, hr):
        if random.random() < 0.5:
            lr = lr[:, ::-1, :].copy()
            hr = hr[:, ::-1, :].copy()
        return lr, hr

    @staticmethod
    def random_vertical_flip(lr, hr):
        if random.random() < 0.5:
            lr = lr[::-1, :, :].copy()
            hr = hr[::-1, :, :].copy()
        return lr, hr

    @staticmethod
    def random_rotate_90(lr, hr):
        if random.random() < 0.5:
            lr = np.rot90(lr, axes=(1, 0)).copy()
            hr = np.rot90(hr, axes=(1, 0)).copy()
        return lr, hr

    def __getitem__(self, idx):
        lr_image = Image.open(self.dataset[idx]['lr']).convert('RGB')
        hr_image = resize_image(lr_image, Image.open(self.dataset[idx]['hr']).convert('RGB'), scale=self.scale)
        lr = np.array(lr_image)
        hr = np.array(hr_image)
        lr, hr = self.random_crop(lr, hr, self.patch_size, self.scale)
        lr, hr = self.random_horizontal_flip(lr, hr)
        lr, hr = self.random_vertical_flip(lr, hr)
        lr, hr = self.random_rotate_90(lr, hr)
        lr = lr.astype(np.float32).transpose([2, 0, 1]) / 255.0
        hr = hr.astype(np.float32).transpose([2, 0, 1]) / 255.0
        return lr, hr

    def __len__(self):
        return len(self.dataset)


class TrainAugmentDatasetH5(Dataset):
    def __init__(self, h5_file, scale, patch_size=64):
        super(TrainAugmentDatasetH5, self).__init__()
        self.h5_file = h5_file
        self.patch_size = patch_size
        self.scale = scale

    @staticmethod
    def random_crop(lr, hr, size, scale):
        lr_left = random.randint(0, lr.shape[1] - size)
        lr_right = lr_left + size
        lr_top = random.randint(0, lr.shape[0] - size)
        lr_bottom = lr_top + size
        hr_left = lr_left * scale
        hr_right = lr_right * scale
        hr_top = lr_top * scale
        hr_bottom = lr_bottom * scale
        lr = lr[lr_top:lr_bottom, lr_left:lr_right]
        hr = hr[hr_top:hr_bottom, hr_left:hr_right]
        return lr, hr

    @staticmethod
    def random_horizontal_flip(lr, hr):
        if random.random() < 0.5:
            lr = lr[:, ::-1, :].copy()
            hr = hr[:, ::-1, :].copy()
        return lr, hr

    @staticmethod
    def random_vertical_flip(lr, hr):
        if random.random() < 0.5:
            lr = lr[::-1, :, :].copy()
            hr = hr[::-1, :, :].copy()
        return lr, hr

    @staticmethod
    def random_rotate_90(lr, hr):
        if random.random() < 0.5:
            lr = np.rot90(lr, axes=(1, 0)).copy()
            hr = np.rot90(hr, axes=(1, 0)).copy()
        return lr, hr

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr = f['lr'][str(idx)][::]
            hr = f['hr'][str(idx)][::]
            lr, hr = self.random_crop(lr, hr, self.patch_size, self.scale)
            lr, hr = self.random_horizontal_flip(lr, hr)
            lr, hr = self.random_vertical_flip(lr, hr)
            lr, hr = self.random_rotate_90(lr, hr)
            lr = lr.astype(np.float32).transpose([2, 0, 1]) / 255.0
            hr = hr.astype(np.float32).transpose([2, 0, 1]) / 255.0
            return lr, hr

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])
