from typing import List

import torch as ch
import torchvision
import numpy as np 
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter
from torchvision import datasets, transforms
# Note that statistics are wrt to uin8 range, [0,255].
CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]

def get_cifar10_loaders(batch_size,subset=None):
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='/tmldata1/fdangelo/understanding-weight-decay/data/CIFAR', train=True, download=True, transform=transform_train)
    if subset is not None:
        # Create a new dataset with 10000 training points and 1000 points per class
        new_train_data = []
        new_train_labels = []
        for i in range(10):
            indices = np.where(np.array(trainset.targets) == i)[0][:subset]
            new_train_data.append(trainset.data[indices])
            new_train_labels += [i]*subset

        new_train_data = np.concatenate(new_train_data, axis=0)
        trainset.data = new_train_data
        trainset.targets = new_train_labels

    trainloader = ch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=20)

    testset = torchvision.datasets.CIFAR10(root='/tmldata1/fdangelo/understanding-weight-decay/data/CIFAR', train=False, download=True, transform=transform_test)
    testloader = ch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=20)

    loaders = {'train': trainloader, 'test': testloader}
    return loaders

def create_dataloaders(dataset, no_data_augm , batch_size, device):
    loaders = {}
    for name in ['train', 'test']:
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

        # Add image transforms and normalization
        if name == 'train' and not no_data_augm:
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(padding=2),
                Cutout(8, tuple(map(int, CIFAR_MEAN))), # Note Cutout is done before normalization.
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice(device, non_blocking=True),
            ToTorchImage(),
            Convert(ch.float16),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        # Create loaders
        if dataset == 'cifar10_binary': 
            path = f'/tmldata1/fdangelo/understanding-weight-decay/data/CIFAR/cifar_10000'
        elif dataset == 'cifar10_10000':
            path = f'/tmldata1/fdangelo/understanding-weight-decay/data/CIFAR/cifar_10000'
        elif dataset == 'cifar10_5k':
            path = f'/tmldata1/fdangelo/understanding-weight-decay/data/CIFAR/cifar10_5k'
        elif dataset == 'cifar100':
            path = f'/tmldata1/fdangelo/understanding-weight-decay/data/CIFAR100/cifar100'
        else: 
            path = f'/tmldata1/fdangelo/understanding-weight-decay/data/CIFAR/cifar'
        loaders[name] = Loader(path+f'_{name}.beton',
                                batch_size=batch_size,
                                num_workers=20,
                                order=OrderOption.RANDOM,
                                drop_last=(name == 'train'),
                                pipelines={'image': image_pipeline,
                                           'label': label_pipeline})
    return loaders 


shapes_dict = {'mnist': (60000, 1, 28, 28),
               'mnist_binary': (13007, 1, 28, 28),
               'svhn': (73257, 3, 32, 32),
               'cifar10': (50000, 3, 32, 32),
               'cifar10_horse_car': (10000, 3, 32, 32),
               'cifar10_binary': (10000, 3, 32, 32),
               'cifar10_10000': (10000, 3, 32, 32),
               'cifar10_dog_cat': (10000, 3, 32, 32),
               'cifar100': (50000, 3, 32, 32),
               'uniform_noise': (1000, 1, 28, 28),
               'gaussians_binary': (1000, 1, 1, 100),
               }

datasets_dict = {'mnist': datasets.MNIST,
                 'mnist_binary': datasets.MNIST,
                 'svhn': datasets.SVHN,
                 'cifar10': datasets.CIFAR10,
                 'cifar10_horse_car': datasets.CIFAR10,
                 'cifar10_binary': datasets.CIFAR10,
                 'cifar10_10000': datasets.CIFAR10,
                 'cifar10_dog_cat': datasets.CIFAR10,
                 'cifar100': datasets.CIFAR100,
                 }

classes_dict = {'cifar10': {0: 'airplane',
                            1: 'automobile',
                            2: 'bird',
                            3: 'cat',
                            4: 'deer',
                            5: 'dog',
                            6: 'frog',
                            7: 'horse',
                            8: 'ship',
                            9: 'truck',
                            }
                }


                