from typing import List
import os 
import torch as ch
import torchvision
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
import numpy as np 
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter
from torchvision import datasets
# Note that statistics are wrt to uin8 range, [0,255].
CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]

def create_dataloaders(dataset, no_data_augm , batch_size, device):
    loaders = {}

    # Create loaders
    if dataset == 'cifar10': 
        path = f'data/CIFAR10/cifar10'
        if not os.path.isfile(path+f'_train.beton'): 
            data = {
                'train': torchvision.datasets.CIFAR10('data/CIFAR10', train=True, download=True),
                'test': torchvision.datasets.CIFAR10('data/CIFAR10', train=False, download=True)}
            for (name, ds) in data.items():
                writer = DatasetWriter(f'data/CIFAR10/cifar10_{name}.beton', {
                    'image': RGBImageField(),
                    'label': IntField()
                })
                writer.from_indexed_dataset(ds)

    elif dataset == 'cifar10_5k':
        path = f'data/CIFAR10/cifar10_5k'
        if not os.path.isfile(path+f'_train.beton'): 
            data = {
                'train': torchvision.datasets.CIFAR10('data/CIFAR10', train=True, download=True),
                'test': torchvision.datasets.CIFAR10('data/CIFAR10', train=False, download=True)}
            n_points = 500 #n_points per class 
            new_train_data = []
            new_train_labels = []
            for i in range(10):
                indices = np.where(np.array(data['train'].targets) == i)[0][:n_points]
                new_train_data.append(data['train'].data[indices])
                new_train_labels += [i]*n_points
            new_train_data = np.concatenate(new_train_data, axis=0)
            data['train'].data = new_train_data
            data['train'].targets = new_train_labels
            for (name, ds) in data.items():
                writer = DatasetWriter(f'data/CIFAR10/cifar10_5k_{name}.beton', {
                    'image': RGBImageField(),
                    'label': IntField()
                })
                writer.from_indexed_dataset(ds)

    elif dataset == 'cifar100':
        path = f'data/CIFAR100/cifar100'
        if not os.path.isfile(path+f'_train.beton'): 
            data = {
                'train': torchvision.datasets.CIFAR10('data/CIFAR10', train=True, download=True),
                'test': torchvision.datasets.CIFAR10('data/CIFAR10', train=False, download=True)}
            for (name, ds) in data.items():
                writer = DatasetWriter(f'data/CIFAR100/cifar100_{name}.beton', {
                    'image': RGBImageField(),
                    'label': IntField()
                })
                writer.from_indexed_dataset(ds)



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


                