# Copyright (c) OpenMMLab. All rights reserved.
import math
import random

import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler

from mmcls.datasets import SAMPLERS


@SAMPLERS.register_module()
class GroupSampler(Sampler):

    def __init__(self, dataset,
                 samples_per_gpu,
                 num_replicas=None,
                 rank=None,
                 shuffle=True):
        # print("neizhishuxing")
        # print( '\n'.join(['%s:%s' % item for item in dataset.__dict__.items()]))
        assert hasattr(dataset,'flag')  
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)  # 每个正整数的出现次数, 从0开始
        self.num_samples = 0
        for i, size in enumerate(self.group_sizes):
            self.num_samples += int(np.ceil(
                size / self.samples_per_gpu)) * self.samples_per_gpu
        self.group_arrow = np.where(self.flag == 1)[0]
        self.group_others = np.where(self.flag == 0)[0]

    # def __iter__(self):
    #     indices = []
    #     for i, size in enumerate(self.group_sizes):
    #         if size == 0:
    #             continue
    #         indice = np.where(self.flag == i)[0]
    #         assert len(indice) == size
    #         np.random.shuffle(indice)
    #         num_extra = int(np.ceil(size / self.samples_per_gpu)
    #                         ) * self.samples_per_gpu - len(indice)
    #         indice = np.concatenate(
    #             [indice, np.random.choice(indice, num_extra)])
    #         indices.append(indice)
    #     indices = np.concatenate(indices)  # 按组concatenate，先训练一组，再训练另一组
    #     indices = [
    #         indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
    #         for i in np.random.permutation(
    #             range(len(indices) // self.samples_per_gpu))
    #     ]
    #     indices = np.concatenate(indices)
    #     indices = indices.astype(np.int64).tolist()
    #     assert len(indices) == self.num_samples
    #     return iter(indices)

    def __iter__(self):
        indices = []
        for i in range(int(self.num_samples / self.samples_per_gpu)):
            arrow_count = int(np.random.choice([0.2,0.3]) * self.samples_per_gpu)
            arrow_indice = np.random.choice(self.group_arrow, arrow_count)

            indice = np.concatenate(
                [arrow_indice, np.random.choice(self.group_others, self.samples_per_gpu - arrow_count)])
            indices.append(indice)
        indices = np.concatenate(indices)  # 按组concatenate，先训练一组，再训练另一组
        indices = [
            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(
                range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        seed (int, optional): random seed used to shuffle the sampler if
            ``shuffle=True``. This number should be identical across all
            processes in the distributed group. Default: 0.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 seed=0):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed if seed is not None else 0

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                # add .numpy() to avoid bug when selecting indice in parrots.
                # TODO: check whether torch.randperm() can be replaced by
                # numpy.random.permutation().
                indice = indice[list(
                    torch.randperm(int(size), generator=g).numpy())].tolist()
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                # pad indice
                tmp = indice.copy()
                for _ in range(extra // size):
                    indice.extend(tmp)
                indice.extend(tmp[:extra % size])
                indices.extend(indice)

        assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch