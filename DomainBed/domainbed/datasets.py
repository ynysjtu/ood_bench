# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os, pickle
from io import BytesIO

import torch
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate

import numpy as np
import scipy.stats as stats
import pandas as pd
from PIL import Image, ImageFile

from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Extra-small images (14x14)
    "ColoredMNIST_IRM",
    "ColoredMNIST_IRM_IID",
    "ColoredMNIST_IRM_Blue",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    "NICOAnimal",
    "NICOVehicle",
    "NICOMixed",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW",
    # ImageNet variants
    "ImageNet_R",
    "ImageNet_A",
    "ImageNet_V2",
    "ImageNetSketch",
    # CelebA splits
    "CelebA_Blond",
    # CUB
    "CUB_200_bill_shape",
    "CUB_200_wing_color",
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)
    
    def get_transform(self, input_size, normalize, scheme):
        if scheme == 'domainbed':
            augment_transform = transforms.Compose([
                # transforms.Resize((224,224)),
                transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                normalize
            ])
        
        elif scheme == 'jigen':
            augment_transform = transforms.Compose([
                # transforms.Resize((224,224)),
                transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                normalize
            ])
        elif scheme == 'decaug_nico':
            augment_transform = transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        elif scheme == 'jigen_wo_color_aug':
            augment_transform = transforms.Compose([
                # transforms.Resize((224,224)),
                transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            raise NotImplementedError
            
        return augment_transform


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                         self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()
    
    
class ColoredMNIST_IRM(MultipleDomainDataset):
    N_STEPS = 5001
    CHECKPOINT_FREQ = 500
    ENVIRONMENTS = ['+90%', '+80%', '-90%']
    INPUT_SHAPE = (2, 14, 14)

    def __init__(self, root, test_envs, hparams):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)

        original_images = original_dataset_tr.train_data
        original_labels = original_dataset_tr.train_labels

        shuffle = torch.randperm(len(original_images))
        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        environments = (0.1, 0.2, 0.9)
        for i, env in enumerate(environments[:-1]):
            images = original_images[:50000][i::2]
            labels = original_labels[:50000][i::2]
            self.datasets.append(self.color_dataset(images, labels, env))
        images = original_images[50000:]
        labels = original_labels[50000:]
        self.datasets.append(self.color_dataset(images, labels, environments[-1]))

        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # Subsample 2x for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()
    
    
class ColoredMNIST_IRM_IID(MultipleDomainDataset):
    N_STEPS = 5001
    CHECKPOINT_FREQ = 500
    ENVIRONMENTS = ['+50%', '+50%']
    INPUT_SHAPE = (2, 14, 14)

    def __init__(self, root, test_envs, hparams):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)

        original_images = original_dataset_tr.train_data
        original_labels = original_dataset_tr.train_labels

        shuffle = torch.randperm(len(original_images))
        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        environments = [0.5, 0.5]
        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(self.color_dataset(images, labels, environments[i]))

        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # Subsample 2x for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()
    
    
class ColoredMNIST_IRM_Blue(MultipleDomainDataset):
    N_STEPS = 5001
    CHECKPOINT_FREQ = 500
    ENVIRONMENTS = ['train_env1', 'test_env']
    INPUT_SHAPE = (3, 14, 14)

    def __init__(self, root, test_envs, hparams):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(f'{root}', train=True, download=False)
        original_images = original_dataset_tr.train_data
        original_labels = original_dataset_tr.train_labels

        shuffle = torch.randperm(len(original_images))
        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        environments = hparams['cmnist_env_ps']
        blue_means = hparams['cmnist_blue_means']
        blue_stds = hparams['cmnist_blue_stds']
        # environments = [0.0, 1.0]
        # backgrounds = [0.0, float(hparams['colored_mnist_background'])]
        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(self.color_dataset(images, labels, environments[i], blue_means[i], blue_stds[i]))

        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2

    def color_dataset(self, images, labels, environment, blue_mean, blue_std):
        # Subsample 2x for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment, len(labels)))
        images = torch.stack([images, images, torch.zeros_like(images)], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0
        
        if blue_std > 0:
            scale = blue_std
            a, b = (0 - blue_mean) / scale, (1 - blue_mean) / scale
            intensity = stats.truncnorm.rvs(a, b, loc=blue_mean, scale=scale, size=images.size(0))
            intensity = torch.tensor(intensity)
            dimmed_images = (images.float() * (1 - intensity.view(-1, 1, 1, 1)).float())
            dimmed_images[:, 2] = (images.sum(1).float() * intensity.view(-1, 1, 1).float())
            images = dimmed_images
        else:
            images = images.float()
            images[:, :2] *= (1 - blue_mean)
            images[:, 2] += torch.ones_like(images[:, 2]) * blue_mean
        
#             import matplotlib
#             matplotlib.use('Agg')
#             import matplotlib.pyplot as plt
#             plt.hist(intensity, density=True)
#             plt.savefig(f'tmp/env{environment}_blue{blue_mean}_hist.png')
#             plt.clf()

#             for i in range(5):
#                 t = images[i].data.numpy().astype(np.uint8).transpose(1, 2, 0)
#                 Image.fromarray(t).save(f'tmp/env{environment}_blue{blue_mean}_sample{i}.png')
            
#             t = torch.where(mask[i, 0] > 200, torch.zeros_like(images[i, 0]), torch.ones_like(images[i, 0]) * intensity[i])
#             Image.fromarray(t.data.numpy()).save(f'tmp/env{environment}_bg{background}_sample{i}_gray.png')

#             Image.fromarray(torch.cat([images[i], t.unsqueeze(0)], dim=0).data.numpy().transpose(1, 2, 0)).save(f'tmp/env{environment}_bg{background}_sample{i}.png')

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                                               resample=Image.BICUBIC)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)

class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir() and not f.name.startswith('.')]
        environments = sorted(environments)

        normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize
        ])

        augment_transform = self.get_transform(
            224, normalize, hparams.get('data_augmentation_scheme', 'domainbed'))

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)

class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/kfold/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "DomainNet/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "TerraIncognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)
        
        
class NICOMixedEnvironment(torch.utils.data.Dataset):
    def __init__(self, images_root, csv_file_path, input_shape, transform):
        super().__init__()
        self.label_dict = {'animal': 0, 'vehicle': 1}
        self.transform = transform
        self.img_paths = []
        self.targets = []
        with open(csv_file_path) as f:
            for line in f.readlines():
                img_path, category_name, context_name, superclass = line.strip().split(',')
                img_path = img_path.replace('\\', '/')
                self.img_paths.append(f'{images_root}/{superclass}/images/{img_path}')
                self.targets.append(self.label_dict[superclass])
                
    def __len__(self):
        return len(self.targets)
                
    def __getitem__(self, key):
        with open(self.img_paths[key], 'rb') as f:
            image = Image.open(f).convert('RGB')
            image = self.transform(image)
        return image, self.targets[key]
    
    
class NICOMixed(MultipleDomainDataset):
#     ENVIRONMENTS = ["train1", "train2", "train3", "train4", "val", "test"]
    ENVIRONMENTS = ["train1", "train2", "val", "test"]
    CHECKPOINT_FREQ = 200
    def __init__(self, root, test_envs, hparams):
        self.input_shape = (3, 224, 224)
        self.datasets = []
        self.num_classes = 2
        
        normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        transform = transforms.Compose([
            transforms.Resize((int(self.input_shape[1] / 0.875), int(self.input_shape[2] / 0.875))),
            transforms.CenterCrop(self.input_shape[1]),
            transforms.ToTensor(),
            normalize
        ])

        augment_transform = self.get_transform(
            self.input_shape[1], normalize, hparams.get('data_augmentation_scheme', 'domainbed'))
        
        for i, env_name in enumerate(self.ENVIRONMENTS):
            if hparams['data_augmentation'] and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform
            csv_file_path = os.path.join(f'{root}/NICO/mixed_split/env_{env_name}.csv')
            self.datasets.append(NICOMixedEnvironment(f'{root}/NICO', csv_file_path, self.input_shape, env_transform))
        
        
class NICOEnvironment(torch.utils.data.Dataset):
    def __init__(self, images_root, csv_file_path, input_shape, transform, category_dict):
        super().__init__()
        self.transform = transform
        
        self.img_paths = []
        self.targets = []
        with open(csv_file_path) as f:
            for line in f.readlines():
                img_path, category_name, context_name = line.strip().split(',')[:3]
                img_path = img_path.replace('\\', '/')
                self.img_paths.append(f'{images_root}/{img_path}')
                self.targets.append(category_dict[category_name])
                
    def __len__(self):
        return len(self.targets)
                
    def __getitem__(self, key):
        with open(self.img_paths[key], 'rb') as f:
            image = Image.open(f).convert('RGB')
            image = self.transform(image)
        return image, self.targets[key]
    
        
class NICOMultipleDomainDataset(MultipleDomainDataset):
#     ENVIRONMENTS = [f"train{i}" for i in range(20)] + ["val", "test"]
#     N_WORKERS = 2
    ENVIRONMENTS = ["domain_1", "domain_2", "domain_3", "domain_4", "domain_val"]
    CHECKPOINT_FREQ = 1000
    def __init__(self, root, superclass, test_envs, category_dict, hparams):
        self.input_shape = (3, 224, 224)
        self.datasets = []
        
        if superclass == 'animal':
            normalize = transforms.Normalize(
                    mean=[0.408, 0.421, 0.412], std=[0.186, 0.191, 0.209])
        else:
            normalize = transforms.Normalize(
                    mean=[0.624, 0.609, 0.607], std=[0.220, 0.219, 0.211])
        
        transform = transforms.Compose([
            transforms.Resize((int(self.input_shape[1] / 0.875), int(self.input_shape[2] / 0.875))),
            transforms.CenterCrop(self.input_shape[1]),
            transforms.ToTensor(),
            normalize
        ])

        augment_transform = self.get_transform(
            self.input_shape[1], normalize, hparams.get('data_augmentation_scheme', 'domainbed'))
            
        split_name = hparams['nico_split_name']
        
        for i, env_name in enumerate(self.ENVIRONMENTS):
            if hparams['data_augmentation'] and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform
            csv_file_path = os.path.join(f'/home/ma-user/work/IIRM/NICO_settings/{split_name}/{superclass}_{env_name}.csv')
            self.datasets.append(NICOEnvironment(f'{root}/nico/{superclass}/images', csv_file_path, self.input_shape, env_transform, category_dict))
            
            
class NICOAnimal(NICOMultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        image_folders = ['bear', 'bird', 'cat', 'cow', 'dog', 'elephant', 'horse', 'monkey', 'rat', 'sheep']
        self.num_classes = len(image_folders)
        # NOTE: use the following for the original classes:
        # category_dict = {name: i for i, name in enumerate(image_folders)}
        # NOTE: use the following for 2 classes:
        category_dict = {'0': 0, '1': 1}
        super().__init__(root, 'animal', test_envs, category_dict, hparams)
            

class NICOVehicle(NICOMultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        image_folders = ['airplane', 'bicycle', 'boat', 'bus', 'car', 'helicopter', 'motorcycle', 'train', 'truck']
        self.num_classes = len(image_folders)
        category_dict = {name: i for i, name in enumerate(image_folders)}
        super().__init__(root, 'vehicle', test_envs, category_dict, hparams)

            

class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform
        self.targets = self.dataset.y_array[self.indices]

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)
    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()
        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

        normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])

        augment_transform = self.get_transform(
            self.input_shape[1], normalize, hparams.get('data_augmentation_scheme', 'domainbed'))

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)


    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = [ "hospital_0", "hospital_1", "hospital_2", "hospital_3",
            "hospital_4"]
    CHECKPOINT_FREQ = 1000
    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=f'{root}/WILDS')
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = [ "region_0", "region_1", "region_2", "region_3",
            "region_4", "region_5"]
    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=f'{root}/WILDS')
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)
        
class ImageNetVariant(MultipleDomainDataset):
    def __init__(self, root, environments, test_envs, augment, hparams):
        super().__init__()
        print(environments)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)
        
        
class ImageNet_R(ImageNetVariant):
    ENVIRONMENTS = ["imagenet", "imagenet-r"]
    def __init__(self, root, test_envs, hparams):
        super().__init__(root, ['imagenet-subset-r200/train', 'imagenet-r'],
                         test_envs, hparams['data_augmentation'], hparams)

        
class ImageNet_A(ImageNetVariant):
    ENVIRONMENTS = ["imagenet", "imagenet-a"]
    def __init__(self, root, test_envs, hparams):
        super().__init__(root, ['imagenet-subset-a200/train', 'imagenet-a'],
                         test_envs, hparams['data_augmentation'], hparams)
        

class ImageNet_V2(ImageNetVariant):
    ENVIRONMENTS = ["imagenet", "imagenet-v2"]
    def __init__(self, root, test_envs, hparams):
        super().__init__(root, ['ILSVRC/Data/CLS-LOC/train', 'imagenetv2-matched-frequency-format-val'],
                         test_envs, hparams['data_augmentation'], hparams)
        
        
class ImageNetSketch(ImageNetVariant):
    ENVIRONMENTS = ["imagenet", "imagenet-sketch"]
    def __init__(self, root, test_envs, hparams):
        super().__init__(root, ['ILSVRC/Data/CLS-LOC/train', 'imagenet-sketch'],
                         test_envs, hparams['data_augmentation'], hparams)

        
# this class is adapted from https://github.com/chingyaoc/fair-mixup/blob/master/celeba/main_dp.py
class CelebA(torch.utils.data.Dataset):
    def __init__(self, dataframe, folder_dir, target_id, transform=None, cdiv=0, ccor=0):
        self.dataframe = dataframe
        self.folder_dir = folder_dir
        self.target_id = target_id
        self.transform = transform
        self.file_names = dataframe.index
        self.targets = np.concatenate(dataframe.labels.values).astype(int)
        gender_id = 20
        
        target_idx0 = np.where(self.targets[:, target_id] == 0)[0]
        target_idx1 = np.where(self.targets[:, target_id] == 1)[0]
        gender_idx0 = np.where(self.targets[:, gender_id] == 0)[0]
        gender_idx1 = np.where(self.targets[:, gender_id] == 1)[0]
        nontarget_males = list(set(gender_idx1) & set(target_idx0))
        nontarget_females = list(set(gender_idx0) & set(target_idx0))
        target_males = list(set(gender_idx1) & set(target_idx1))
        target_females = list(set(gender_idx0) & set(target_idx1))
        
        u1 = len(nontarget_males) - int((1 - ccor) * (len(nontarget_males) - len(nontarget_females)))
        u2 = len(target_females) - int((1 - ccor) * (len(target_females) - len(target_males)))
        selected_idx = nontarget_males[:u1] + nontarget_females + target_males + target_females[:u2]
        self.targets = self.targets[selected_idx]
        self.file_names = self.file_names[selected_idx]
        
        target_idx0 = np.where(self.targets[:, target_id] == 0)[0]
        target_idx1 = np.where(self.targets[:, target_id] == 1)[0]
        gender_idx0 = np.where(self.targets[:, gender_id] == 0)[0]
        gender_idx1 = np.where(self.targets[:, gender_id] == 1)[0]
        nontarget_males = list(set(gender_idx1) & set(target_idx0))
        nontarget_females = list(set(gender_idx0) & set(target_idx0))
        target_males = list(set(gender_idx1) & set(target_idx1))
        target_females = list(set(gender_idx0) & set(target_idx1))
        
        selected_idx = nontarget_males + nontarget_females[:int(len(nontarget_females) * (1 - cdiv))] + target_males + target_females[:int(len(target_females) * (1 - cdiv))]
        self.targets = self.targets[selected_idx]
        self.file_names = self.file_names[selected_idx]
        
        target_idx0 = np.where(self.targets[:, target_id] == 0)[0]
        target_idx1 = np.where(self.targets[:, target_id] == 1)[0]
        gender_idx0 = np.where(self.targets[:, gender_id] == 0)[0]
        gender_idx1 = np.where(self.targets[:, gender_id] == 1)[0]
        nontarget_males = list(set(gender_idx1) & set(target_idx0))
        nontarget_females = list(set(gender_idx0) & set(target_idx0))
        target_males = list(set(gender_idx1) & set(target_idx1))
        target_females = list(set(gender_idx0) & set(target_idx1))
        print(len(nontarget_males), len(nontarget_females), len(target_males), len(target_females))
                        
        self.targets = self.targets[:, self.target_id]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.folder_dir, self.file_names[index]))
        label = self.targets[index]
        if self.transform:
            image = self.transform(image)
        return image, label


class CelebA_Blond(MultipleDomainDataset):
    ENVIRONMENTS = ["unbalanced_1", "unbalanced_2", "balanced"]
    N_STEPS = 2001
    CHECKPOINT_FREQ = 200
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        environments = self.ENVIRONMENTS
        print(environments)
        
        self.input_shape = (3, 224, 224,)
        self.num_classes = 2 # blond or not
        
        dataframes = []
        for env_name in ('tr_env1', 'tr_env2', 'te_env'):
            with open(f'{root}/celeba/blond_split/{env_name}_df.pickle', 'rb') as handle:
                dataframes.append(pickle.load(handle))
        tr_env1, tr_env2, te_env = dataframes
        
        orig_w = 178
        orig_h = 218
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        images_path = f'{root}/celeba/img_align_celeba'
        transform = transforms.Compose([
            transforms.CenterCrop(min(orig_w, orig_h)),
            transforms.Resize(self.input_shape[1:]),
            transforms.ToTensor(),
            normalize,
        ])

        if hparams['data_augmentation']:
            augment_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.input_shape[1:],
                                             scale=(0.7, 1.0), ratio=(1.0, 1.3333333333333333)),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.0),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            
            if hparams.get('test_data_augmentation', False):
                transform = augment_transform
        else:
            augment_transform = transform
        
        cdiv = hparams.get('cdiv', 0)
        ccor = hparams.get('ccor', 1)
        
        target_id = 9
        tr_dataset_1 = CelebA(pd.DataFrame(tr_env1), images_path, target_id, transform=augment_transform,
                              cdiv=cdiv, ccor=ccor)
        tr_dataset_2 = CelebA(pd.DataFrame(tr_env2), images_path, target_id, transform=augment_transform,
                              cdiv=cdiv, ccor=ccor)
        te_dataset = CelebA(pd.DataFrame(te_env), images_path, target_id, transform=transform)
        
        self.datasets = [tr_dataset_1, tr_dataset_2, te_dataset]
        
        
class CUB_200(torch.utils.data.Dataset):
    def __init__(self, image_root, annos_file, attr_root, ids_file, attr_id, transform=None):
        """
        :param image_root: images dir root
        :param annos_file:
        :param attr_root: attributes dir root
        :param ids_file: 'train_id.txt' or 'test_id.txt'
        :param transform: Image Transform
        :param attr_ids: attributes selected
        """
        super(CUB_200, self).__init__()

        self.image_root = image_root
        self.images_name = []
        self.targets = []

        with open(f'{attr_root}/images.txt') as f:
            for line in f.readlines():
                image_id, image_name = line.strip().split()
                image_name = image_name.split('/')[-1]
                self.images_name.append(image_name)
        assert len(self.images_name) == 11788

        with open(ids_file) as f:
            ids = [int(s) - 1 for s in f.readlines()]

        attr_mat = np.zeros((11788, 312), dtype=np.int)
        with open(f'{attr_root}/image_attribute_labels.txt') as f:
            for line in f.readlines():
                attr = line.strip('\n').split(' ')  # image_id, attribute_id, is_present, certainty_id, time
                if int(attr[3]) >= 3 and int(attr[2]) == 1:
                    attr_mat[int(attr[0])-1][int(attr[1])-1] = 1

        self.images_name = [self.images_name[id] for id in ids]
        self.targets = attr_mat[ids, attr_id - 1].tolist()

        self.classes = set(self.targets)

        if transform is None:
            self.transform = T.Compose([T.ToTensor()])
        else:
            self.transform = transform

    def __getitem__(self, index):
        """
        :param index: int
        :return: image: Tensor: (3, w, h)
                 label: str
                 attributes: List[], length equals to attr_ids in __init__ function: e.g. [0, 1, 1]
        """

        image_name = self.images_name[index]

        image = Image.open(os.path.join(self.image_root, image_name)).convert('RGB')
        image = self.transform(image)

        label = self.targets[index]

        return image, label

    def __len__(self):
        return len(self.images_name)


class CUB_200_normal(MultipleDomainDataset):
    ENVIRONMENTS = ["env0", "env1"]
    N_STEPS = 2001
    CHECKPOINT_FREQ = 200
    def __init__(self, root, test_envs, hparams):
        assert test_envs == [1]

        normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize
        ])

        augment_transform = self.get_transform(
            224, normalize, hparams.get('data_augmentation_scheme', 'domainbed'))

        tr_transform = augment_transform if hparams.get('data_augmentation', False) else transform
        te_transform = transform

        tr_dataset = CUB_200(f'{root}/CUB_200/images', 'cub200_dataset/annotation.csv', 'cub200_dataset/attributes', 'cub200_dataset/train_id.txt', 7, transform=tr_transform)
        te_dataset = CUB_200(f'{root}/CUB_200/images', 'cub200_dataset/annotation.csv', 'cub200_dataset/attributes', 'cub200_dataset/test_id.txt', 7, transform=te_transform)

        self.datasets = [tr_dataset, te_dataset]
        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)
    
    
class CUB_200_bill_shape(MultipleDomainDataset):
    ENVIRONMENTS = ["env0", "env1"]
    N_STEPS = 2001
    CHECKPOINT_FREQ = 200
    def __init__(self, root, test_envs, hparams):
        assert test_envs == [1]
        
        normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize
        ])

        augment_transform = self.get_transform(
            224, normalize, hparams.get('data_augmentation_scheme', 'domainbed'))
        
        tr_transform = augment_transform if hparams.get('data_augmentation', False) else transform
        te_transform = transform

        attr_id = 7  # has_bill_shape::all-purpose
        tr_dataset = CUB_200(f'{root}/CUB_200/images', 'cub200_dataset/annotation.csv', 'cub200_dataset/attributes', 'cub200_dataset/train_id.txt', attr_id, transform=tr_transform)
        te_dataset = CUB_200(f'{root}/CUB_200/images', 'cub200_dataset/annotation.csv', 'cub200_dataset/attributes', 'cub200_dataset/test_id.txt', attr_id, transform=te_transform)
        
        self.datasets = [tr_dataset, te_dataset]
        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)
        
        
class CUB_200_normal(MultipleDomainDataset):
    ENVIRONMENTS = ["env0", "env1"]
    N_STEPS = 2001
    CHECKPOINT_FREQ = 200
    def __init__(self, root, test_envs, hparams):
        assert test_envs == [1]

        normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize
        ])

        augment_transform = self.get_transform(
            224, normalize, hparams.get('data_augmentation_scheme', 'domainbed'))

        tr_transform = augment_transform if hparams.get('data_augmentation', False) else transform
        te_transform = transform

        tr_dataset = CUB_200(f'{root}/CUB_200/images', 'cub200_dataset/annotation.csv', 'cub200_dataset/attributes', 'cub200_dataset/train_id.txt', 7, transform=tr_transform)
        te_dataset = CUB_200(f'{root}/CUB_200/images', 'cub200_dataset/annotation.csv', 'cub200_dataset/attributes', 'cub200_dataset/test_id.txt', 7, transform=te_transform)

        self.datasets = [tr_dataset, te_dataset]
        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)
    
    
class CUB_200_wing_color(MultipleDomainDataset):
    ENVIRONMENTS = ["env0", "env1"]
    N_STEPS = 2001
    CHECKPOINT_FREQ = 200
    def __init__(self, root, test_envs, hparams):
        assert test_envs == [1]
        
        normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize
        ])

        augment_transform = self.get_transform(
            224, normalize, hparams.get('data_augmentation_scheme', 'domainbed'))
        
        tr_transform = augment_transform if hparams.get('data_augmentation', False) else transform
        te_transform = transform

        attr_id = 21  # has_wing_color::black
        tr_dataset = CUB_200(f'{root}/CUB_200/images', 'cub200_dataset/annotation.csv', 'cub200_dataset/attributes', 'cub200_dataset/train_id.txt', attr_id, transform=tr_transform)
        te_dataset = CUB_200(f'{root}/CUB_200/images', 'cub200_dataset/annotation.csv', 'cub200_dataset/attributes', 'cub200_dataset/test_id.txt', attr_id, transform=te_transform)
        
        self.datasets = [tr_dataset, te_dataset]
        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)
