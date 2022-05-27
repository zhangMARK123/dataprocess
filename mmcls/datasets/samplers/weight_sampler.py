import pandas as pd
import torch
import torch.utils.data
import torchvision
from torch.utils.data.sampler import Sampler
import numpy as np
from mmcls.datasets import SAMPLERS


@SAMPLERS.register_module()
class ImbalancedDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, num_replicas=None, rank=None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["color_labels"], df["shape_labels"], df["toward_labels"], df["character_labels"], df[
            "simplelights_labels"] = dataset.get_sampler_labels()
        df["contact"] = df["color_labels"].apply(str) + df["shape_labels"].apply(str) + df["toward_labels"].apply(str) + \
                        df["character_labels"].apply(str) + df["simplelights_labels"]
        count = df['contact'].value_counts()
        weights = count[df["contact"]].to_numpy()
        weights[np.where(weights == 0)] = 0.3

        weights = 1 / weights
        weights[np.where(weights > 1)] = 0

        f = open("weights.txt", 'w')
        tempweight = ",".join(str(x) for x in weights)
        f.write(tempweight)
        f.close()
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
