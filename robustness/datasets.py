"""
Module containing all the supported datasets, which are subclasses of the
abstract class :class:`robustness.datasets.DataSet`. 

Currently supported datasets:

- ImageNet (:class:`robustness.datasets.ImageNet`)
- RestrictedImageNet (:class:`robustness.datasets.RestrictedImageNet`)
- CIFAR-10 (:class:`robustness.datasets.CIFAR`)
- CINIC-10 (:class:`robustness.datasets.CINIC`)
- A2B: horse2zebra, summer2winter_yosemite, apple2orange
  (:class:`robustness.datasets.A2B`)

:doc:`../example_usage/training_lib_part_2` shows how to add custom
datasets to the library.
"""

import os

import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
import torch as ch
import torch.utils.data
import torchvision
from torch import nn
from torch.utils.data import Dataset
from functools import partial
from collections import OrderedDict

from . import imagenet_models, cifar_models
from torchvision import transforms, datasets

from .tools import constants
from . import data_augmentation as da
from . import loaders

from .tools.helpers import get_label_mapping

###
# Datasets: (all subclassed from dataset)
# In order:
## ImageNet
## Restricted Imagenet 
## Other Datasets:
## - CIFAR
## - CINIC
## - A2B (orange2apple, horse2zebra, etc)
###

class DataSet(object):
    '''
    Base class for representing a dataset. Meant to be subclassed, with
    subclasses implementing the `get_model` function. 
    '''

    def __init__(self, ds_name, data_path, **kwargs):
        """
        Args:
            ds_name (str) : string identifier for the dataset
            data_path (str) : path to the dataset 
            num_classes (int) : *required kwarg*, the number of classes in
                the dataset
            mean (ch.tensor) : *required kwarg*, the mean to normalize the
                dataset with (e.g.  :samp:`ch.tensor([0.4914, 0.4822,
                0.4465])` for CIFAR-10)
            std (ch.tensor) : *required kwarg*, the standard deviation to
                normalize the dataset with (e.g. :samp:`ch.tensor([0.2023,
                0.1994, 0.2010])` for CIFAR-10)
            custom_class (type) : *required kwarg*, a
                :samp:`torchvision.models` class corresponding to the
                dataset, if it exists (otherwise :samp:`None`)
            label_mapping (dict[int,str]) : *required kwarg*, a dictionary
                mapping from class numbers to human-interpretable class
                names (can be :samp:`None`)
            transform_train (torchvision.transforms) : *required kwarg*, 
                transforms to apply to the training images from the
                dataset
            transform_test (torchvision.transforms) : *required kwarg*,
                transforms to apply to the validation images from the
                dataset
        """
        required_args = ['num_classes', 'mean', 'std', 'custom_class',
            'label_mapping', 'transform_train', 'transform_test']
        assert set(kwargs.keys()) == set(required_args), "Missing required args, only saw %s" % kwargs.keys()
        self.ds_name = ds_name
        self.data_path = data_path
        self.__dict__.update(kwargs)

    def get_model(self, arch, pretrained):
        '''
        Should be overriden by subclasses. Also, you will probably never
        need to call this function, and should instead by using
        `model_utils.make_and_restore_model </source/robustness.model_utils.html>`_.

        Args:
            arch (str) : name of architecture 
            pretrained (bool): whether to try to load torchvision 
                pretrained checkpoint

        Returns:
            A model with the given architecture that works for each
            dataset (e.g. with the right input/output dimensions).
        '''

        raise NotImplementedError

    def make_loaders(self, workers, batch_size, data_aug=True, subset=None,
                 subset_start=0, subset_type='rand', val_batch_size=None,
                 only_val=False, shuffle_train=True, shuffle_val=False):
        '''
        Args:
            workers (int) : number of workers for data fetching (*required*).
                batch_size (int) : batch size for the data loaders (*required*).
            data_aug (bool) : whether or not to do train data augmentation.
            subset (None|int) : if given, the returned training data loader
                will only use a subset of the training data; this should be a
                number specifying the number of training data points to use.
            subset_start (int) : only used if `subset` is not None; this specifies the
                starting index of the subset.
            subset_type ("rand"|"first"|"last") : only used if `subset is
                not `None`; "rand" selects the subset randomly, "first"
                uses the first `subset` images of the training data, and
                "last" uses the last `subset` images of the training data.
            seed (int) : only used if `subset == "rand"`; allows one to fix
                the random seed used to generate the subset (defaults to 1).
            val_batch_size (None|int) : if not `None`, specifies a
                different batch size for the validation set loader.
            only_val (bool) : If `True`, returns `None` in place of the
                training data loader
            shuffle_train (bool) : Whether or not to shuffle the training data
                in the returned DataLoader.
            shuffle_val (bool) : Whether or not to shuffle the test data in the
                returned DataLoader.

        Returns:
            A training loader and validation loader according to the
            parameters given. These are standard PyTorch data loaders, and
            thus can just be used via:

            >>> train_loader, val_loader = ds.make_loaders(workers=8, batch_size=128) 
            >>> for im, lab in train_loader:
            >>>     # Do stuff...
        '''
        transforms = (self.transform_train, self.transform_test)
        return loaders.make_loaders(workers=workers,
                                    batch_size=batch_size,
                                    transforms=transforms,
                                    data_path=self.data_path,
                                    data_aug=data_aug,
                                    dataset=self.ds_name,
                                    label_mapping=self.label_mapping,
                                    custom_class=self.custom_class,
                                    val_batch_size=val_batch_size,
                                    subset=subset,
                                    subset_start=subset_start,
                                    subset_type=subset_type,
                                    only_val=only_val,
                                    shuffle_train=shuffle_train,
                                    shuffle_val=shuffle_val)

class MNIST(DataSet):
    def __init__(self, data_path, **kwargs):
        ds_kwargs = {
            'num_classes': 10,
            'mean': ch.tensor([0.1307]),
            'std': ch.tensor([0.3081]),
            'custom_class': datasets.MNIST,
            'label_mapping': None,
            'transform_train': transforms.Compose([torchvision.transforms.ToTensor()]),   # da.TRAIN_TRANSFORMS_DEFAULT(28),
            'transform_test': transforms.Compose([torchvision.transforms.ToTensor()])   # da.TEST_TRANSFORMS_DEFAULT(28),
        }
        super(MNIST, self).__init__('mnist', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained):
        if pretrained:
            raise ValueError("MNIST does not support pytorch_pretrained=True")
        return cifar_models.__dict__[arch]()

def add_background(image):
    """
    Takes a processed image of 224x224 and place it on a 448x448 background

    - image (torch.Tensor) of size 224x224

    Returns:
    - upscaled_image (torch.Tensor) of size 448x448
    """
    sz = 224
    
    # pick a random position to place the image
    indices = torch.randint(0, sz+1, (2,))
    x = indices[0].item()
    y = indices[1].item()

    upscaled_image = torch.zeros((3, 448, 448))
    upscaled_image[..., x:x+sz, y:y+sz] = image

    return upscaled_image

class HAM10000(DataSet):
    def __init__(self, data_path, file_name, label_mapping, custom_class, \
                 apply_ablation=False, saliency_dir=None, perc_ablation=0, **kwarg):
        """
        Args:
        - data_path (str): path to folder with the dataset
        - file_name (str): the CSV file keeping the dataset split.
                It is passed as parameter to HAM10000_dataset
        - label_mapping (OrderedDict): mapping between label id and class
                OrderedDict([
                    (0, 'bkl'),
                    (1, 'nv'),
                    (2, 'vasc')
                ])
        - apply_ablation(boolean): If `True`, then don't apply transforms.CenterCrop and transforms.Resize
                because they were already applied when retrieving the image
        """
        # Compute number of classes from the csv
        self.num_classes = len(list(label_mapping.keys()))
        self.saliency_dir = saliency_dir
        self.perc_ablation = perc_ablation

        input_size = 224

        # Use imagenet mean and std, because I use a pre-trained model.
        imagenet_mean = [0.485, 0.456, 0.406] # RGB
        imagenet_std = [0.229, 0.224, 0.225]

        train_transforms, test_transforms = [], []
        if apply_ablation == False: 
            """
            If `apply_ablation` is True, then these transforms are 
            """
            train_transforms.extend([
                transforms.CenterCrop(450), 
                transforms.Resize((input_size, input_size))])

            test_transforms.extend([
                transforms.CenterCrop(450),
                transforms.Resize((input_size, input_size))
            ])

        train_transforms.extend([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(10),
            transforms.RandomRotation(50),
            transforms.ToTensor()
        ])
        TRAIN_TRANSFORMS_HAM10000 = transforms.Compose(train_transforms)
        
        test_transforms.extend([transforms.ToTensor()])
        TEST_TRANSFORMS_HAM10000 = transforms.Compose(test_transforms)

        ds_kwargs = {
            'num_classes': self.num_classes,
            'mean': ch.tensor(imagenet_mean), # Just add again the mean and std. I didn't find where this is used.
            'std': ch.tensor(imagenet_std),
            'custom_class': partial(custom_class, file_name=file_name, 
                                    apply_ablation=apply_ablation, saliency_dir=saliency_dir, perc_ablation=perc_ablation),
            'label_mapping': label_mapping,
            'transform_train': TRAIN_TRANSFORMS_HAM10000,
            'transform_test': TEST_TRANSFORMS_HAM10000
        }
        super().__init__('ham10000', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained=True, custom_head=None):
        """
        - custom_head (nn.Module) - custom head architecture to add after the convolutional layers
        """
        # if arch not in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
            # raise ValueError("Currently HAM10000 supports only Resnet")

        # The model is initialized from an ImageNet model, which has 1000 classes
        # Thus, I need to replace the last layer to have `self.num_classes` logits

        model = imagenet_models.__dict__[arch](num_classes=1000, pretrained=pretrained)
        freeze(model)
        num_ftrs = model.fc.in_features

        if custom_head:
            model.fc = custom_head
        else:
            fc_activation = nn.ReLU()

            model.fc = nn.Sequential(
                nn.Dropout(p=0.25),
                nn.Linear(num_ftrs, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(512, self.num_classes)
            )
        
        # Initialize the weights in the head
        for m in model.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        return model

class HAM10000_3cls(HAM10000):
    def __init__(self, data_path, file_name, apply_ablation=False, saliency_dir=None, perc_ablation=0,
                 use_dropout_head=False, dropout_perc=0, **kwarg):
        self.use_dropout_head = use_dropout_head
        self.dropout_perc = dropout_perc
        
        label_mapping = OrderedDict([
            (0, 'nv'),
            (1, 'mel'),
            (2, 'bkl')
        ])

        self.num_classes = len(label_mapping)
        super().__init__(data_path, file_name, label_mapping, custom_class=HAM10000_dataset_3cls_balanced,
                         apply_ablation=apply_ablation, saliency_dir=saliency_dir, perc_ablation=perc_ablation, **kwarg)

    def get_model(self, arch='resnet18', pretrained=True):
        if arch == 'resnet50':
            prev_size = 2048
        else:
            prev_size = 512

        if self.use_dropout_head:
                custom_head = nn.Sequential(
                    nn.Linear(prev_size, 512),
                    nn.ReLU(),
                    nn.BatchNorm1d(512),
                    nn.Dropout(self.dropout_perc),
                    nn.Linear(512, self.num_classes)
                )
        else:
            custom_head = nn.Sequential(
                nn.Linear(prev_size, self.num_classes)
            )

        return super().get_model(arch, pretrained=pretrained, custom_head=custom_head)

class ImageNet(DataSet):
    '''
    ImageNet Dataset [DDS+09]_.

    Requires ImageNet in ImageFolder-readable format. 
    ImageNet can be downloaded from http://www.image-net.org. See
    `here <https://pytorch.org/docs/master/torchvision/datasets.html#torchvision.datasets.ImageFolder>`_
    for more information about the format.

    .. [DDS+09] Deng, J., Dong, W., Socher, R., Li, L., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. 2009 IEEE Conference on Computer Vision and Pattern Recognition, 248-255.

    '''
    def __init__(self, data_path, **kwargs):
        """
        """
        ds_kwargs = {
            'num_classes': 1000,
            'mean': ch.tensor([0.485, 0.456, 0.406]),
            'std': ch.tensor([0.229, 0.224, 0.225]),
            'custom_class': None,
            'label_mapping': None,
            'transform_train': da.TRAIN_TRANSFORMS_IMAGENET,
            'transform_test': da.TEST_TRANSFORMS_IMAGENET
        }
        super(ImageNet, self).__init__('imagenet', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained):
        """
        """
        return imagenet_models.__dict__[arch](num_classes=self.num_classes,
                                        pretrained=pretrained)

class RestrictedImageNet(DataSet):
    '''
    RestrictedImagenet Dataset [TSE+19]_

    A subset of ImageNet with the following labels:

    * Dog (classes 151-268)
    * Cat (classes 281-285)
    * Frog (classes 30-32)
    * Turtle (classes 33-37)
    * Bird (classes 80-100)
    * Monkey (classes 365-382)
    * Fish (classes 389-397)
    * Crab (classes 118-121)
    * Insect (classes 300-319)

    To initialize, just provide the path to the full ImageNet dataset
    (no special formatting required).

    .. [TSE+19] Tsipras, D., Santurkar, S., Engstrom, L., Turner, A., &
        Madry, A. (2019). Robustness May Be at Odds with Accuracy. ICLR
        2019.
    '''
    def __init__(self, data_path, **kwargs):
        """
        """
        ds_name = 'restricted_imagenet'
        ds_kwargs = {
            'num_classes': len(constants.RESTRICTED_IMAGNET_RANGES),
            'mean': ch.tensor([0.4717, 0.4499, 0.3837]),
            'std': ch.tensor([0.2600, 0.2516, 0.2575]),
            'custom_class': None,
            'label_mapping': get_label_mapping(ds_name,
                constants.RESTRICTED_IMAGNET_RANGES),
            'transform_train': da.TRAIN_TRANSFORMS_IMAGENET,
            'transform_test': da.TEST_TRANSFORMS_IMAGENET
        }
        super(RestrictedImageNet, self).__init__(ds_name,
                data_path, **ds_kwargs)

    def get_model(self, arch, pretrained):
        """
        """
        if pretrained:
            raise ValueError("Dataset doesn't support pytorch_pretrained")
        return imagenet_models.__dict__[arch](num_classes=self.num_classes)

class CustomImageNet(DataSet):
    '''
    CustomImagenet Dataset 

    A subset of ImageNet with the user-specified labels

    To initialize, just provide the path to the full ImageNet dataset
    along with a list of lists of wnids to be grouped together
    (no special formatting required).

    '''
    def __init__(self, data_path, custom_grouping, **kwargs):
        """
        """
        ds_name = 'custom_imagenet'
        ds_kwargs = {
            'num_classes': len(custom_grouping),
            'mean': ch.tensor([0.4717, 0.4499, 0.3837]),
            'std': ch.tensor([0.2600, 0.2516, 0.2575]),
            'custom_class': None,
            'label_mapping': get_label_mapping(ds_name,
                custom_grouping),
            'transform_train': da.TRAIN_TRANSFORMS_IMAGENET,
            'transform_test': da.TEST_TRANSFORMS_IMAGENET
        }
        super(CustomImageNet, self).__init__(ds_name,
                data_path, **ds_kwargs)

    def get_model(self, arch, pretrained):
        """
        """
        if pretrained:
            raise ValueError("Dataset doesn't support pytorch_pretrained")
        return imagenet_models.__dict__[arch](num_classes=self.num_classes)

class CIFAR(DataSet):
    """
    CIFAR-10 dataset [Kri09]_.

    A dataset with 50k training images and 10k testing images, with the
    following classes:

    * Airplane
    * Automobile
    * Bird
    * Cat
    * Deer
    * Dog
    * Frog
    * Horse
    * Ship
    * Truck

    .. [Kri09] Krizhevsky, A (2009). Learning Multiple Layers of Features
        from Tiny Images. Technical Report.
    """
    def __init__(self, data_path='/tmp/', **kwargs):
        """
        """
        ds_kwargs = {
            'num_classes': 10,
            'mean': ch.tensor([0.4914, 0.4822, 0.4465]),
            'std': ch.tensor([0.2023, 0.1994, 0.2010]),
            'custom_class': datasets.CIFAR10,
            'label_mapping': None,
            'transform_train': da.TRAIN_TRANSFORMS_DEFAULT(32),
            'transform_test': da.TEST_TRANSFORMS_DEFAULT(32)
        }
        super(CIFAR, self).__init__('cifar', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained):
        """
        """
        return cifar_models.__dict__[arch](num_classes=self.num_classes)

class CINIC(DataSet):
    """
    CINIC-10 dataset [DCA+18]_.

    A dataset with the same classes as CIFAR-10, but with downscaled images
    from various matching ImageNet classes added in to increase the size of
    the dataset.

    .. [DCA+18] Darlow L.N., Crowley E.J., Antoniou A., and A.J. Storkey
        (2018) CINIC-10 is not ImageNet or CIFAR-10. Report
        EDI-INF-ANC-1802 (arXiv:1810.03505)
    """
    def __init__(self, data_path, **kwargs):
        """
        """
        ds_kwargs = {
            'num_classes': 10,
            'mean': ch.tensor([0.47889522, 0.47227842, 0.43047404]),
            'std': ch.tensor([0.24205776, 0.23828046, 0.25874835]),
            'custom_class': None,
            'label_mapping': None,
            'transform_train': da.TRAIN_TRANSFORMS_DEFAULT(32),
            'transform_test': da.TEST_TRANSFORMS_DEFAULT(32)
        }
        super(CINIC, self).__init__('cinic', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained):
        """
        """
        if pretrained:
            raise ValueError('CINIC does not support pytorch_pretrained=True')
        return cifar_models.__dict__[arch](num_classes=self.num_classes)

class A2B(DataSet):
    """
    A-to-B datasets [ZPI+17]_

    A general class for image-to-image translation dataset. Currently
    supported are:
    
    * Horse <-> Zebra
    * Apple <-> Orange
    * Summer <-> Winter

    .. [ZPI+17] Zhu, J., Park, T., Isola, P., & Efros, A.A. (2017).
        Unpaired Image-to-Image Translation Using Cycle-Consistent
        Adversarial Networks. 2017 IEEE International Conference on
        Computer Vision (ICCV), 2242-2251.
    """
    def __init__(self, data_path, **kwargs):
        """
        """
        _, ds_name = os.path.split(data_path)
        valid_names = ['horse2zebra', 'apple2orange', 'summer2winter_yosemite']
        assert ds_name in valid_names, \
                f"path must end in one of {valid_names}, not {ds_name}"
        ds_kwargs = {
            'num_classes': 2,
            'mean': ch.tensor([0.5, 0.5, 0.5]),
            'custom_class': None,
            'std': ch.tensor([0.5, 0.5, 0.5]),
            'transform_train': da.TRAIN_TRANSFORMS_IMAGENET,
            'label_mapping': None,
            'transform_test': da.TEST_TRANSFORMS_IMAGENET
        }
        super(A2B, self).__init__(ds_name, data_path, **ds_kwargs)

    def get_model(self, arch, pretrained=False):
        """
        """
        if pretrained:
            raise ValueError('A2B does not support pytorch_pretrained=True')
        return imagenet_models.__dict__[arch](num_classes=self.num_classes)

DATASETS = {
    'imagenet': ImageNet,
    'restricted_imagenet': RestrictedImageNet,
    'custom_imagenet': CustomImageNet,
    'cifar': CIFAR,
    'cinic': CINIC,
    'a2b': A2B,
    'mnist': MNIST,
    'ham10000': HAM10000
}
'''
Dictionary of datasets. A dataset class can be accessed as:

>>> import robustness.datasets
>>> ds = datasets.DATASETS['cifar']('/path/to/cifar')
'''


def upsample_dataframe(df):
    """
    Given a dataframe, upsample the items to have equal number of items in each class
    """
    df_upsampled = df.iloc[0:0]

    # compute the maximum count
    counts = df['dx'].value_counts().to_dict()
    maxx = np.max(list(counts.values()))

    # multiply the df that many times
    for lesion, count in counts.items():
        upsampling_factor = int(np.ceil(maxx / count))

        # upsample
        aux = df.iloc[0:0]
        for i in range(upsampling_factor):
            aux = aux.append(df.loc[df['dx'] == lesion])

        # drop the last rows exceeding the max count
        aux.reset_index(inplace=True, drop=True)
        aux.drop(np.arange(maxx, aux.shape[0]), inplace=True)

        df_upsampled = df_upsampled.append(aux)

    df_upsampled.reset_index(inplace=True, drop=True)

    return df_upsampled

# Custom Dataset to load my datasets
class HAM10000_dataset(Dataset):
    def __init__(self, root, file_name, train=True, download=None, transform=None, upsample=True, test=False):
        """
        Gets called in `loaders.make_loaders` with exactly these parameters.

        - test (bool): If `True`, then load the test set!

        """
        self.transform = transform
        self.file_name = file_name

        data = pd.read_csv(os.path.join(root, file_name))
        if test:
            self.df = data.loc[data['split'] == 'test']
        else:
            if train:
                self.df = data.loc[data['split'] == 'train']
            else:
                self.df = data.loc[data['split'] == 'valid']

        self.df.reset_index(inplace=True)

        if upsample:
            self.df = upsample_dataframe(self.df)

        self.df['path'] = root + '/' + self.df.loc[:, 'path_relative']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        Load data and get label
        """
        X, y = self._getitem_index(index)
        y = torch.tensor(y)

        if self.transform:
            X = self.transform(X)

        return X, y

    def _getitem_index(self, index):
        """
        Get item without applying any transform.
        """
        X = Image.open(self.df['path'][index])
        y = int(self.df['type'][index])

        return X, y

    def _getitem_image_id(self, image_id):
        row = self.df[self.df.image_id == image_id]
        X = Image.open(row['path'].iloc[0])
        y = int(row['type'].iloc[0])

        return X, y

def ablate_features(image, mask, ablation_type='white'):
    """
    Ablate an image, based on a given mask.
    
    Args:
    - image (torch.Tensor of (3, 224, 224))
    - mask (binary torch.Tensor of (3, 224, 224)): Contains value of 1 of the feature which should be removed
    """
    # remove ablated feature by making them white
    ablated_image = image * (ch.tensor(1)-mask) + mask

    if ablation_type == 'mean':
        # compute the mean
        mean0 = image[0, :, :].mean()
        mean1 = image[1, :, :].mean()
        mean2 = image[2, :, :].mean()

        ablated_image[0, :, :] = ablated_image[0, :, :] + mean0 * mask[0, :, :]
        ablated_image[1, :, :] = ablated_image[1, :, :] + mean1 * mask[1, :, :]
        ablated_image[2, :, :] = ablated_image[2, :, :] + mean2 * mask[2, :, :]

    return ablated_image

def get_mask(saliency_map_2d, perc=0.3):
    """
    Given a saliency map, return a mask corresponding to a percentage of the top values.

    Args:
    - saliency_map (np.ndarray (224, 224))
    - perc (float): percentage of saliency to remove e.g. 0.3

    Returns:
    - mask (binary torch.Tensor(3, 224, 224))
    """
    pixels_sorted = np.sort(saliency_map_2d.reshape(-1)) # sort in ascending order
    pixels_sorted = pixels_sorted[::-1]      # reverse the array

    # Pick the kth value percentage-wise from the top
    kth_value_index = int(224*224*perc-1)
    kth_value = pixels_sorted[kth_value_index]
    kth_value

    # Create mask
    mask = saliency_map_2d >= kth_value # (224, 224)
    mask = ch.tensor(mask.astype(np.int))
    mask = mask.repeat(3, 1, 1) # (3, 224, 224)

    return mask

def save_saliency_map(map, DATA_DIR, model_name, image_id):
    """
    Given a mask for abalting feature, save it in the director 'DATA_DIR/saliency_maps/model_name'

    Args:
    - mask (torch.Tensor (3, 224, 224)) - binary tensor with values 0, 1
    - DATA_DIR (str) - path to data directory
    - model_name (str) - used to create the folder for storing the files 
    - image_id (str) - the id of the image for which the mask was generated
    """
    dir_path = os.path.join(DATA_DIR, 'saliency_maps', model_name)

    if os.path.exists(dir_path) == False:
        os.mkdir(dir_path)

    ch.save(map, os.path.join(dir_path, image_id))

class HAM10000_dataset_3cls_balanced(HAM10000_dataset):
    def __init__(self, root, file_name, train=True, download=None, transform=None, upsample=True, test=False, 
                 apply_ablation=False, saliency_dir=None, perc_ablation=0):
        """
        Gets called in `loaders.make_loaders` with exactly these parameters.

        This class represents the 3_cls_balanced, which is creted to work for cross_validation!
        The dataset has 5 folds. This class should work for both when this is the internal training
        split (4 fold), internal validation (1 fold) and the test set.
        
        - root (str): root path
        - file_name (str): Contains the file name (e.g. `data.csv`), followed by "::" and the 
                id for the VALIDATION fold! 
                (note that when test=True, appending "::...") does not change anything

                e.g. "data.csv::5" with flag `train=True` means self.df contains folds 1, 2, 3, 4
                e.g. "data.csv::5" with flag `train=False` means self.df contains folds 5
                e.g. "data.csv" with flag 'test=True' means self.df contains the entire dataset

        - test (bool): If `True`, then load the test set from `root`
        - apply_ablation (boolean): If `True`, then load and apply the saliency map
        - saliency_dir (str): Path to the directory from where to load and apply the saliency mask
        """
        if apply_ablation and saliency_dir == None:
            raise ValueError("If apply_ablation==True, then you must provide `saliency_dir`")

        self.transform = transform
        self.file_name = file_name
        self.apply_ablation = apply_ablation
        self.saliency_dir = saliency_dir
        self.perc_ablation = perc_ablation

        aux = file_name.split('::')
        data = pd.read_csv(os.path.join(root, aux[0]))
        if test == True:
            self.df = data
        else: # train/validation
            aux = aux[1]
            if aux == '0': # train on the whole training set
                if train == True:
                    self.df = data[data['fold'].isin(['1', '2', '3', '4', '5'])]
                else:
                    self.df = data[data['fold'].isin({'validation'})]
            else:
                train_folds = {'1', '2', '3', '4', '5'}
                train_folds.remove(aux)

                if train == True:
                    self.df = data[data['fold'].isin(train_folds)]
                else:
                    self.df = data[data['fold'].isin([aux])]

        if upsample:
            self.df = upsample_dataframe(self.df)

        print(f"Created dataset of length: {len(self.df)}")
        self.df = self.df.reset_index()
        self.df['path'] = root + '/' + self.df.loc[:, 'path_relative']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        Load data and get label
        """
        X, y = self._getitem_index(index)

        y = torch.tensor(y)

        if self.apply_ablation:
            # load saliency
            image_id = self.df['image_id'][index]
            saliency_map_2d = torch.load(os.path.join(
                self.saliency_dir, image_id))  # np.ndarray of (224, 224)
            mask = get_mask(saliency_map_2d, perc=self.perc_ablation)
    
            X = transforms.Compose([
                transforms.CenterCrop(450), 
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                partial(ablate_features, mask=mask),
                transforms.ToPILImage()
            ])(X)

        if self.transform:
            X = self.transform(X)

        return X, y

    def _getitem_index(self, index):
        """
        Get item without applying any transform.
        """
        X = Image.open(self.df['path'][index])
        y = int(self.df['type'][index])

        return X, y

    def _getitem_image_id(self, image_id):
        row = self.df[self.df.image_id == image_id]
        X = Image.open(row['path'].iloc[0])
        y = int(row['type'].iloc[0])

        return X, y


def freeze(model):
    """
    Function to `freeze` all weight in the model
    """
    for param in model.parameters():
        param.requires_grad = False

def unfreeze(model, until=0):
    """
    Unfreezes blocks of a ResNet starting from the head to the first layer.

    Args:
    - model (robustness/cifar_models/resnet.py, which is instance of nn.Module) - instace of ResNet-18 model
    - until (int) - the layer until to unfreeze
        - 5: model.fc
        - 4: model.fc + model.layer4
        - 3: model.fc + model.layer4 + model.layer3
        - 2: model.fc + model.layer4 + model.layer3 + model.layer2
        - 1: (full unfreeze)
    """
    assert until in [1, 2, 3, 4, 5], ("Paramter 'until' needs to have values in [1, 2, 3, 4, 5]")

    if until == 1:
        for param in model.parameters():
            param.requires_grad = True
        print("Unfrozen the entire model")
        return

    if until<=5:
        for param in model.fc.parameters():
            param.requires_grad = True
        print("Unfrozen layer .fc")
    if until<=4:
        for param in model.layer4.parameters():
            param.requires_grad = True
        print("Unfrozen layer .layer4")
    if until<=3:
        for param in model.layer3.parameters():
            param.requires_grad = True
        print("Unfrozen layer .layer3")
    if until<=2:
        for param in model.layer2.parameters():
            param.requires_grad = True
        print("Unfrozen layer .layer2")
    
