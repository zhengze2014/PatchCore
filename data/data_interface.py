import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from data.mvtecDataset import MVTecDataset


class DInterface(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.dataset_path = kwargs['dataset_path']
        self.category = kwargs['category']
        self.load_size = kwargs['load_size']
        self.input_size = kwargs['input_size']
        self.batch_size = kwargs['batch_size']
        self.num_workers = kwargs['num_workers']
        # imageNet
        self.mean_train = [0.485, 0.456, 0.406]
        self.std_train = [0.229, 0.224, 0.225]

        self.data_transforms = transforms.Compose([
            transforms.Resize((self.load_size, self.load_size)),
            transforms.ToTensor(),
            transforms.CenterCrop(self.input_size),
            transforms.Normalize(mean=self.mean_train,
                                 std=self.std_train)])
        self.gt_transforms = transforms.Compose([
            transforms.Resize((self.load_size, self.load_size)),
            transforms.ToTensor(),
            transforms.CenterCrop(self.input_size)])

    def train_dataloader(self):
        image_datasets = MVTecDataset(root=os.path.join(self.dataset_path, self.category),
                                      transform=self.data_transforms,
                                      gt_transform=self.gt_transforms,
                                      phase='train')
        train_loader = DataLoader(image_datasets, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers)
        return train_loader

    def test_dataloader(self):
        test_datasets = MVTecDataset(root=os.path.join(self.dataset_path, self.category),
                                     transform=self.data_transforms,
                                     gt_transform=self.gt_transforms,
                                     phase='test')
        test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False,
                                 num_workers=self.num_workers)  # , pin_memory=True) # only work on batch_size=1, now.

        return test_loader


