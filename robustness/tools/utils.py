import torch
import random
import numpy as np

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensor = tensor.clone()
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def fix_random_seed(seed):
    """
      Fix random seed to get a deterministic output
      Inputs:
      - seed_no: seed number to be fixed
    """
    # torch.manual_seed(seed)
    # random.seed(seed)
    # np.random.seed(seed)

    # TODO - The seed was changed on 7th June - some results use the above seed setting.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
