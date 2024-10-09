import torch 
import torchvision
import numpy as np

image_size = 224


transforms_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(brightness = (0.8, 1.3))], p=0.5),
    torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(contrast = (0.8, 1.3))], p=0.5),
    #torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(saturation = (0.5, 1.5))], p=0.5),   # Skin color?
    #torchvision.transforms.RandomApply([torchvision.transforms.Grayscale()], p = 0.1),
    torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(kernel_size = 11)], p = 0.2),
    torchvision.transforms.Resize(image_size),
    torchvision.transforms.CenterCrop(image_size)
    #torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transforms_val_test = torchvision.transforms.Compose([
     torchvision.transforms.Resize(image_size),
     torchvision.transforms.CenterCrop(image_size)
     #torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class CustomDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""
    def __init__(self, img_ten, label_ten, transforms=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_ten = img_ten
        self.label_ten = label_ten
        self.transforms = transforms

        assert len(img_ten) == len(label_ten), "img_ten and label ten must have equal size"
    def __len__(self):
      return len(self.img_ten)

    def __getitem__(self, idx):
      x = self.img_ten[idx]
      y = self.label_ten[idx]

      if not self.transforms == None:
        x = self.transforms(x)

      return x, y

    def set_transforms(self, transforms):
      self.transforms = transforms



class CustomDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""
    def __init__(self, img_ten, label_ten, transforms=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_ten = img_ten
        self.label_ten = label_ten
        self.transforms = transforms

        assert len(img_ten) == len(label_ten), "img_ten and label ten must have equal size"
    def __len__(self):
      return len(self.img_ten)

    def __getitem__(self, idx):
      x = self.img_ten[idx]
      y = self.label_ten[idx]

      if not self.transforms == None:
        x = self.transforms(x)

      return x, y

    def set_transforms(self, transforms):
      self.transforms = transforms