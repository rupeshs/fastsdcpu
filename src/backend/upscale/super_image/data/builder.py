"""
The DatasetBuilder class, to prepare datasets for image-super-resolution training and evaluation.
The augmentation method and code is from the BAM paper by Wang et al. (2021) at
https://github.com/dandingbudanding/BAM.
"""

import h5py
import glob
from PIL import Image
import numpy as np

from tqdm.auto import tqdm

from torchvision.transforms import transforms


class DatasetBuilder:
    """Class for building training and evaluation datasets.
    `DatasetBuilder` has 1 key method:
        - :meth:`super_image.DatasetBuilder.prepare`: Prepares the source data from a folder and saves to a h5 file.
    """

    @staticmethod
    def prepare(
        base_path: str = None,
        output_path: str = None,
        scale: int = 4,
        do_augmentation: bool = False,
    ):
        """Prepares the source data from a folder and saves to a h5 file.
        Args:
            base_path (str): base path for the folder where the source data is.
            output_path (str): output path for the folder where the source data is.
            scale (int): LR scale to downsize the HR images of the dataset, defaults to 4.
            do_augmentation (bool): Boolean value indicating whether to augment or not, training sets can be augmented.
        """

        h5_file = h5py.File(output_path, 'w')

        lr_group = h5_file.create_group('lr')
        hr_group = h5_file.create_group('hr')

        image_list = sorted(glob.glob(f'{base_path}/*'))
        idx = 0

        if len(image_list) > 0:
            for image_path in tqdm(image_list):
                hr = Image.open(image_path).convert('RGB')

                if do_augmentation:
                    for hr in transforms.FiveCrop(size=(hr.height // 2, hr.width // 2))(hr):
                        hr = hr.resize(((hr.width // scale) * scale, (hr.height // scale) * scale),
                                       resample=Image.BICUBIC)
                        lr = hr.resize((hr.width // scale, hr.height // scale), resample=Image.BICUBIC)

                        hr = np.array(hr)
                        lr = np.array(lr)

                        lr_group.create_dataset(str(idx), data=lr)
                        hr_group.create_dataset(str(idx), data=hr)

                        idx += 1
                else:
                    hr = Image.open(image_path).convert('RGB')
                    hr_width = (hr.width // scale) * scale
                    hr_height = (hr.height // scale) * scale
                    hr = hr.resize((hr_width, hr_height), resample=Image.BICUBIC)
                    lr = hr.resize((hr.width // scale, hr_height // scale), resample=Image.BICUBIC)

                    hr = np.array(hr)
                    lr = np.array(lr)

                    lr_group.create_dataset(str(idx), data=lr)
                    hr_group.create_dataset(str(idx), data=hr)

                    idx += 1
        else:
            raise FileNotFoundError('There are no files in the folder.')

        h5_file.close()
