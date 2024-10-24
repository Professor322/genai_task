from torch.utils.data import Dataset
from utils.data_utils import make_dataset
from utils.class_registry import ClassRegistry
import os
import glob
import numpy as np
import torch
import cv2


datasets_registry = ClassRegistry()


@datasets_registry.add_to_registry(name="base_dataset")
class BaseDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.paths = make_dataset(root)
        self.transforms = transforms

    def __getitem__(self, ind):
        path = self.paths[ind]
        image = Image.open(path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        return {"images": image}

    def __len__(self):
        return len(self.paths)


@datasets_registry.add_to_registry(name="food101_dataset")
class Food101Dataset(Dataset):
    def __init__(self, directory_path, train=True):
        # get classes
        self.dataset_root = directory_path
        self.train = train
        class_names = os.listdir(directory_path)
        class_labels = range(0, len(class_names))
        # for convenience
        self.classes_to_num = {
            class_name: class_label
            for class_name, class_label in zip(class_names, class_labels)
        }
        self.num_to_classes = {
            class_label: class_name
            for class_name, class_label in zip(class_names, class_labels)
        }
        # get paths to the images
        self.image_paths = glob.glob(self.dataset_root + "/*/*")
        print(f"Dataset got: {self.__len__()} images")

    def __getitem__(self, idx):
        # load image and convert it to RGB
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # normalize image to range [-1, 1]
        image = image.astype(np.float32) / 127.5 - 1
        # then convert to torch tensor
        image = torch.tensor(image, dtype=torch.float32)

        # deduct class label from path
        image_class = self.image_paths[idx].split("/")[-2]

        # to comply with improved diffusion unet channels should be
        # at first dimension and labels should be returned as a dict
        return image.permute([2, 0, 1]), {"y": self.classes_to_num[image_class]}

    def __len__(self):
        return len(self.image_paths)


@datasets_registry.add_to_registry(name="fid_dataset")
class ImageFidDataset(Food101Dataset):
    def __init__(self, directory_path, train=True):
        super().__init__(directory_path=directory_path, train=train)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.tensor(image, dtype=torch.uint8).permute([2, 0, 1])
