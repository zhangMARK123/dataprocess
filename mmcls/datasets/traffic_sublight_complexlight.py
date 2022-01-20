import os, json, copy

import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS

from mmcls.core.evaluation import precision_recall_f1, support
from mmcls.models.losses import accuracy


@DATASETS.register_module()
class TrafficSubLightcomplexClsDataset(BaseDataset):

    SHAPE_CLASSES = [
        'circle',
        'uparrow',
        'downarrow',
        'leftarrow',
        'rightarrow',
        'returnarrow',
        'bicycle',   
        'others',
    ]

    COLOR_CLASSES = [
        'red',
        'yellow',
        'green',
        'black',
        'others',
        'unknow'
    
    ]

    TOWARD_CLASSES = [
        'front',
        'side',
        'backside',
        'unknow'
    ]
    CHARACTER_CLASSES=[
        'pass',
        'president',
        'number',
        'word',
        'others',
        'unknow'
    ]
    SIMPLE_CLASSES=['simple',
    'complex']

    def load_annotations(self):
        if self.ann_file is None:
            # load sample from img
            raise (RuntimeError('online sample select nedd ann_file'))
        elif isinstance(self.ann_file, str):
            with open(self.ann_file) as f:
                self.samples = json.load(f)["objects"]
        else:
            raise TypeError('ann_file must be a str or None')

        self.target_samples = []
        for sample_info in self.samples:
            sample_info["img_prefix"] = os.path.join(self.data_prefix, sample_info["data_card_id"])
            ###可在此可以根据各类别比例做标签的更改 先试下缩减类别后的结果，不行就改为多类别
            ###color unknow是指还有子灯的灯箱颜色，但此部分不参加颜色训练。所以此时将颜色4,5,全部归于black不影响训练
            # if sample_info["boxcolor"] == 4 or sample_info["boxcolor"] == 5:
            #     sample_info["boxcolor"] = 3
            ####朝向实际只分了正向侧向，其余样本量为0 所以将此类别归于侧向
            # if sample_info["toward_orientation"] == 2 or sample_info["toward_orientation"] == 3:
            #     sample_info["toward_orientation"] = 1
            if sample_info["bbox"][2] >= 6 and sample_info["bbox"][3] >= 6:
                self.target_samples.append(sample_info)
        return self.target_samples

    @property
    def class_to_idx(self):
        """Map mapping class name to class index.

        Returns:
            dict: mapping from class name to class index.
        """

        return {_class: i for i, _class in enumerate(self.CLASSES)}

    def get_gt_labels(self):
        """Get all ground-truth labels (categories).

        Returns:
            list[int]: categories for all images.
        """
        ###date:1/13 先不管子灯，只写灯箱相关
        color_mask = np.array([data["lightboxcolor_head"] for data in self.data_infos])
        shape_mask = np.array([data["lightboxshape_head"] for data in self.data_infos])
        toward_mask = np.array([data["toward_head"] for data in self.data_infos])
        character_mask = np.array([data["character_head"] for data in self.data_infos])
        simplelight_mask=np.array([data["simplelight_head"] for data in self.data_infos])

        color_labels = np.array([data["boxcolor"] for data in self.data_infos])
        shape_labels = np.array([data["boxshape"] for data in self.data_infos])
        toward_labels = np.array([data['toward_orientation'] for data in self.data_infos])
        character_labels=np.array([data['characteristic'] for data in self.data_infos])
        simplelight_labels=np.array([data["simplelight"] for data in self.data_infos])

        color_labels_selected = color_labels[color_mask]
        shape_labels_selected = shape_labels[shape_mask]
        toward_labels_selected = toward_labels[toward_mask]
        character_labels_selected=character_labels[character_mask]
        simplelight_labels_selected=simplelight_labels[simplelight_mask]
        return color_mask, color_labels_selected, shape_mask, shape_labels_selected, toward_mask, toward_labels_selected,character_mask,character_labels_selected,simplelight_mask,simplelight_labels_selected
    def get_sampler_labels(self):
        """Get all ground-truth labels (categories).

        Returns:
            list[int]: categories for all images.
        """
        color_labels = np.array([data["boxcolor"] for data in self.data_infos])
        shape_labels = np.array([data["boxshape"] for data in self.data_infos])
        toward_labels = np.array([data['toward_orientation'] for data in self.data_infos])
        character_labels=np.array([data['characteristic'] for data in self.data_infos])
        simplelight_labels=np.array([data["simplelight"] for data in self.data_infos])

        return color_labels,shape_labels,toward_labels,character_labels,simplelight_labels
    def get_cat_ids(self, idx):
        """Get category id by index.

        Args:
            idx (int): Index of data.

        Returns:
            int: Image category of specified index.
        """

        return self.data_infos[idx]['gt_label'].astype(np.int)

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        return self.prepare_data(idx)

    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options=None,
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = {'topk': (1, 5)}
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'support'
        ]
        eval_results = {}
        results = np.vstack(results)
        color_mask, color_labels, shape_mask, shape_labels, toward_mask, toward_labels,character_mask,character_labels,simplelight_mask,simplelight_labels = self.get_gt_labels()
        color_result = results[:, :len(self.COLOR_CLASSES)][color_mask]
        shape_result = results[:, len(self.COLOR_CLASSES):len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)][shape_mask]
        toward_result = results[:, len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES):len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)+len(self.TOWARD_CLASSES)][toward_mask]
        character_result = results[:, len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)+len(self.TOWARD_CLASSES):-len(self.SIMPLE_CLASSES)][character_mask]
        simplelight_result = results[:, -len(self.SIMPLE_CLASSES):][simplelight_mask]


        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')

        topk = metric_options.get('topk', (1, 5))
        thrs = metric_options.get('thrs')
        average_mode = metric_options.get('average_mode', 'macro')

        if 'accuracy' in metrics:
            if thrs is not None:
                color_acc = accuracy(color_result, color_labels, topk=topk, thrs=thrs)
                shape_acc = accuracy(shape_result, shape_labels, topk=topk, thrs=thrs)
                toward_acc = accuracy(toward_result, toward_labels, topk=topk, thrs=thrs)
                character_acc=accuracy(character_result,character_labels, topk=topk, thrs=thrs)
                simplelight_acc = accuracy(simplelight_result, simplelight_labels, topk=topk, thrs=thrs)
            else:
                color_acc = accuracy(color_result, color_labels, topk=topk)
                shape_acc = accuracy(shape_result, shape_labels, topk=topk)
                toward_acc = accuracy(toward_result, toward_labels, topk=topk)
                character_acc=accuracy(character_result,character_labels, topk=topk)
                simplelight_acc = accuracy(simplelight_result, simplelight_labels, topk=topk)
            if isinstance(topk, tuple):
                eval_results_ = {
                    f'accuracy_top-{k}': a
                    for k, a in zip(topk, acc)
                }
            else:
                eval_results_ = {'color_accuracy': color_acc, 'shape_accuracy': shape_acc, 'toward_accuracy': toward_acc,'character_accuracy':character_acc,'simplelght_accuracy':simplelight_acc}
            if isinstance(thrs, tuple):
                for key, values in eval_results_.items():
                    eval_results.update({
                        f'{key}_thr_{thr:.2f}': value.item()
                        for thr, value in zip(thrs, values)
                    })
            else:
                eval_results.update(
                    {k: v.item()
                     for k, v in eval_results_.items()})

        return eval_results