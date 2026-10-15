"""Replay buffer, kernel preparation, and back projection extracted from the supplied SR source."""
import random
import numpy as np
import torch
from torch.autograd import Variable
from scipy.ndimage import center_of_mass, shift
from scipy.io import loadmat
from sr_resize import imresize

class ReplayBuffer:

    def __init__(self, max_size=50):
        assert max_size > 0, 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.detach():
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            elif random.uniform(0, 1) > 0.5:
                i = random.randint(0, self.max_size - 1)
                to_return.append(self.data[i].clone())
                self.data[i] = element
            else:
                to_return.append(element)
        return Variable(torch.cat(to_return))

def back_projection(y_sr, y_lr, down_kernel, up_kernel, sf=None):
    y_sr += imresize(y_lr - imresize(y_sr, scale_factor=1.0 / sf, output_shape=y_lr.shape, kernel=down_kernel), scale_factor=sf, output_shape=y_sr.shape, kernel=up_kernel)
    return np.clip(y_sr, 0, 1)

def preprocess_kernels(kernels, conf):
    if kernels is not None:
        if len(kernels) != len(conf.scale_factors):
            raise ValueError('One numeric kernel is required per scale.')
        return [kernel_shift(loadmat(kernel)['Kernel'], sf) for kernel, sf in zip(kernels, conf.scale_factors)]
    else:
        return [conf.downscale_method] * len(conf.scale_factors)

def kernel_shift(kernel, sf):
    current_center_of_mass = center_of_mass(kernel)
    wanted_center_of_mass = np.array(kernel.shape) / 2 + 0.5 * (np.array(sf) - np.array(kernel.shape) % 2)
    shift_vec = wanted_center_of_mass - current_center_of_mass
    kernel = np.pad(kernel, int(np.ceil(np.max(np.abs(shift_vec)))) + 1, 'constant')
    return shift(kernel, shift_vec)
