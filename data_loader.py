import torch

from torch.utils.data import Dataset, DataLoader

import os

from PIL import Image, ImageFile

from utils import image_loader

# the device being on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# try to load truncated images as well
ImageFile.LOAD_TRUNCATED_IMAGES = True

# set max image pixel up
Image.MAX_IMAGE_PIXELS = None


def is_image_file(filename):
    """
    determines whether the arguments is an image file
    @param filename: string of the files path
    @return: a boolean indicating whether the file is an image file
    """
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


class MSCOCODataset(Dataset):
    """
    dataset class for MS-COCO dataset
    """
    def __init__(self, root_dir, loader):
        self.root_dir = root_dir
        self.loader = loader
        self.image_list = [x for x in os.listdir(root_dir) if is_image_file(x)]
        self.len = len(self.image_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = image_loader(img_name, self.loader, add_fake_batch_dimension=False)
        sample = {'image': image}

        return sample


class PainterByNumbersDataset(Dataset):
    """
    dataset class for kaggle painter-by-numbers dataset
    """
    def __init__(self, root_dir, loader):
        self.root_dir = root_dir
        self.loader = loader
        self.image_list = [x for x in os.listdir(root_dir) if is_image_file(x)]
        self.len = len(self.image_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = image_loader(img_name, self.loader, add_fake_batch_dimension=False)
        sample = {'image': image}

        return sample


class MergeCOCOPainterByNumbersDataset(Dataset):
    """
    dataset combining MS-COCO dataset and painter-by-numbers dataset
    """
    def __init__(self, coco_dataset, painter_by_numbers_dataset):
        self.coco_dataset = coco_dataset
        self.painter_by_numbers_dataset = painter_by_numbers_dataset
        self.len = min(self.painter_by_numbers_dataset.__len__(), self.coco_dataset.__len__())

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        coco_data = self.coco_dataset.__getitem__(idx)
        painter_by_numbers_data = self.painter_by_numbers_dataset.__getitem__(idx)
        sample = {'coco': coco_data, 'painter_by_numbers': painter_by_numbers_data}
        return sample


def get_concat_dataloader(configuration):
    """
    creates a dataloader by merging the MS-COCO and the painter-by-numbers dataset
    @param configuration: the configuration file
    @return:
    """
    loader = configuration['loader']
    mscoco_dataset = MSCOCODataset(configuration['coco_data_path'], loader)
    painter_by_numbers_dataset = PainterByNumbersDataset(configuration['painter_by_numbers_data_path'], loader)
    merge_coco_painter_by_numbers_dataset = MergeCOCOPainterByNumbersDataset(mscoco_dataset, painter_by_numbers_dataset)

    concat_dataloader = DataLoader(merge_coco_painter_by_numbers_dataset, batch_size=int(configuration['batch_size']),
                                   shuffle=True, num_workers=16)

    return concat_dataloader
