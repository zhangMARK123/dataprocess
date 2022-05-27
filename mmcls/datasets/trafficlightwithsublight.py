import os, json, copy

import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS

from mmcls.core.evaluation import precision_recall_f1, support
from mmcls.models.losses import accuracy
import random

@DATASETS.register_module()


class TrafficSubLight1ClsDataset(BaseDataset):

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
    NUMBERSUBLIGHT_CLASSES=['3','5']
    def __init__(self,
                 data_prefix,
                 pipeline,
                 classes=None,
                 ann_file=None,
                 test_mode=False):

        super().__init__(
            data_prefix=data_prefix,
            pipeline=pipeline,
            classes=classes,
            ann_file=ann_file,
            test_mode=test_mode)
        if not test_mode:
            self._set_group_flag()
        self.saveflag=0

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        color_hardcase=[]
        shape_hardcase=[]
        f_color=open("/disk3/zs1/mmclassification/projects/complexlight/hard_color.txt","r")
        f_shape=open("/disk3/zs1/mmclassification/projects/complexlight/hard_shape.txt","r")
        for ii in f_color:
            if ii not in color_hardcase:
                color_hardcase.append(ii.strip())
        for jj in f_shape:
            if jj not in shape_hardcase:
                shape_hardcase.append(jj.strip())
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['lightboxshape_head'] and img_info['boxshape'] in [5,6]:
                self.flag[i] = 1
            if img_info['lightboxcolor_head'] and img_info['boxcolor'] in [1]:
                self.flag[i] = 1
            if img_info["data_card_id"]+'/'+img_info["img_info"]["filename"] in color_hardcase:
                self.flag[i] = 1
            if img_info["data_card_id"]+'/'+img_info["img_info"]["filename"] in shape_hardcase:
                self.flag[i] = 1
            if img_info['lightboxshape_head'] and img_info['boxshape'] in [0,1,2,3,4,5,6] and img_info['boxcolor'] in [4]:
                self.flag[i] = 1

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
        color_badcase=[]
        shape_badcase=[]
        f_color=open("/disk3/zs1/mmclassification/projects/complexlight/color.txt","r")
        f_shape=open("/disk3/zs1/mmclassification/projects/complexlight/shape.txt","r")
        for ii in f_color:
            if ii not in color_badcase:
                color_badcase.append(ii.strip())
        for jj in f_shape:
            if jj not in shape_badcase:
                shape_badcase.append(jj.strip())
        for sample_info in self.samples:
            sample_info["img_prefix"] = os.path.join(self.data_prefix, sample_info["data_card_id"])
            if sample_info["ext_occlusion"] == 1 or sample_info["truncation"] == 1:
                continue   
            ###剔除掉badcase
            if sample_info["data_card_id"]+'/'+sample_info["img_info"]["filename"] in color_badcase:
                sample_info["lightboxcolor_head"]=False
            if sample_info["data_card_id"]+'/'+sample_info["img_info"]["filename"] in shape_badcase:
                sample_info["lightboxshape_head"]=False
                


            #筛选正方向灯箱     
            if min(sample_info["bbox"][2], sample_info["bbox"][3])<=0:
                continue
            if max(sample_info["bbox"][2], sample_info["bbox"][3]) / min(sample_info["bbox"][2], sample_info["bbox"][3]) < 2.5:
                continue
            if sample_info["toward_orientation"]==2:
                continue


            ###unknow 类数据不需要这么多
            
            if sample_info["numcolorlight"]>5:
                continue
            if sample_info["boxcolor"]==5:
                sample_info["boxcolor"]=3
            
            ###black 和unknow总数数量较多，减少部分数量
            if random.random()<0.5 and sample_info["boxcolor"]==3:
                continue
            if random.random()<0.6 and sample_info["boxshape"]==7:
                continue
            # if not self.test_mode:
            if sample_info["boxcolor"] in [0,1,2,4] and sample_info["boxshape"]==7:
                sample_info["lightboxshape_head"]=False
            
            ###需要判断下others颜色是否对准确率有影响，若有则舍弃掉
            if sample_info["boxcolor"] in [0,1,2,4] and sample_info["boxshape"]==7 and sample_info["toward_orientation"]==1:
                sample_info["lightboxcolor_head"]=False
                # sample_info["lightboxshape_head"]=False
            if sample_info["boxcolor"]==4:
                sample_info["lightboxcolor_head"]=False
            #无下箭头
            if sample_info["boxshape"]==2:
                sample_info["lightboxshape_head"]=False
            if sample_info["characteristic"]==1 and sample_info["toward_orientation"]==1:
                sample_info["character_head"]=False
            if sample_info["bbox"][2] >=10 and sample_info["bbox"][3] >=10:
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

        numsublight_mask=np.array([data["numsublight_head"] for data in self.data_infos])


        color_labels = np.array([data["boxcolor"] for data in self.data_infos])
        shape_labels = np.array([data["boxshape"] for data in self.data_infos])
        toward_labels = np.array([data['toward_orientation'] for data in self.data_infos])
        character_labels=np.array([data['characteristic'] for data in self.data_infos])
        simplelight_labels=np.array([data["simplelight"] for data in self.data_infos])

        numsublight_labels=np.array([data["numcolorlight"] for data in self.data_infos])

        color_labels_selected = color_labels[color_mask]
        shape_labels_selected = shape_labels[shape_mask]
        toward_labels_selected = toward_labels[toward_mask]
        character_labels_selected=character_labels[character_mask]
        simplelight_labels_selected=simplelight_labels[simplelight_mask]
        
       
        ##复杂子灯
        numsublight_labels_selected=numsublight_labels[numsublight_mask]

        subcolor_mask=np.array([data["sublightcolor_head"] for data in self.data_infos])
       
        subcolor1_label=np.array([data["subcolor0"] for data in self.data_infos])
        
        subcolor2_label=np.array([data["subcolor1"] for data in self.data_infos])
       
        subcolor3_label=np.array([data["subcolor2"] for data in self.data_infos])
        
        subcolor4_label=np.array([data["subcolor3"] for data in self.data_infos])
        
        subcolor5_label=np.array([data["subcolor4"] for data in self.data_infos])
       

        subcolor1_label_selected=subcolor1_label[subcolor_mask]
       
        subcolor2_label_selected=subcolor2_label[subcolor_mask]
        
        subcolor3_label_selected=subcolor3_label[subcolor_mask]
        
        subcolor4_label_selected=subcolor4_label[subcolor_mask]
        
        subcolor5_label_selected=subcolor5_label[subcolor_mask]

        
       
        return color_mask, color_labels_selected, shape_mask, shape_labels_selected, toward_mask, toward_labels_selected,character_mask,character_labels_selected,simplelight_mask,simplelight_labels_selected,numsublight_mask,numsublight_labels_selected, subcolor_mask,subcolor1_label_selected,subcolor2_label_selected,subcolor3_label_selected,subcolor4_label_selected,subcolor5_label_selected
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
        # print(resu)
        scores=np.vstack(results)
        results = np.vstack(results)
        img_metas = self.data_infos
        features=[]
        for i,img_meta in enumerate(img_metas): 
            res={}
            name=img_meta["data_card_id"]+"/"+img_meta["img_info"]["filename"]
            res[name]=str(results[i])
            features.append(res)

        savename1="/disk3/zs1/mmclassification/work_dirs/complexlight_img/onlyhardcase/"+str(self.saveflag)+"hardcase_features.json"
        with open(savename1,'w+') as f:
            f.write(json.dumps({"objects":features}, ensure_ascii=False, indent=4))


       
        color_mask, color_labels, shape_mask, shape_labels, toward_mask, toward_labels,character_mask,character_labels,simplelight_mask,simplelight_labels,numsublight_mask,numsublight_labels,subcolor_mask,subcolor1_labels,subcolor2_labels,subcolor3_labels,subcolor4_labels,subcolor5_labels = self.get_gt_labels()
        color_result = results[:, :len(self.COLOR_CLASSES)][color_mask]
        shape_result = results[:, len(self.COLOR_CLASSES):len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)][shape_mask]
        toward_result = results[:, len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES):len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)+len(self.TOWARD_CLASSES)][toward_mask]
        character_result = results[:, len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)+len(self.TOWARD_CLASSES):len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)+len(self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES)][character_mask]
        simplelight_result = results[:,len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)+len(self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES):len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)+len(self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES)+len(self.SIMPLE_CLASSES)][simplelight_mask]
       
        ##复杂子灯
        numsublight_result=results[:,len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)+len(self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES)+len(self.SIMPLE_CLASSES):len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)+len(self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES)+len(self.SIMPLE_CLASSES)+len(self.NUMBERSUBLIGHT_CLASSES)][numsublight_mask]
        subcolor1_result=results[:,len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)+len(self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES)+len(self.SIMPLE_CLASSES)+len(self.NUMBERSUBLIGHT_CLASSES):len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)+len(self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES)+len(self.SIMPLE_CLASSES)+len(self.NUMBERSUBLIGHT_CLASSES)+len(self.COLOR_CLASSES)][subcolor_mask]
        subcolor2_result=results[:,len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)+len(self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES)+len(self.SIMPLE_CLASSES)+len(self.NUMBERSUBLIGHT_CLASSES)+len(self.COLOR_CLASSES):len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)+len(self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES)+len(self.SIMPLE_CLASSES)+len(self.NUMBERSUBLIGHT_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES)][subcolor_mask]
        subcolor3_result=results[:,len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)+len(self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES)+len(self.SIMPLE_CLASSES)+len(self.NUMBERSUBLIGHT_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES):len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)+len(self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES)+len(self.SIMPLE_CLASSES)+len(self.NUMBERSUBLIGHT_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES)][subcolor_mask]
        subcolor4_result=results[:,len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)+len(self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES)+len(self.SIMPLE_CLASSES)+len(self.NUMBERSUBLIGHT_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES):len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)+len(self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES)+len(self.SIMPLE_CLASSES)+len(self.NUMBERSUBLIGHT_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES)][subcolor_mask]
        subcolor5_result=results[:,len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)+len(self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES)+len(self.SIMPLE_CLASSES)+len(self.NUMBERSUBLIGHT_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES):len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)+len(self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES)+len(self.SIMPLE_CLASSES)+len(self.NUMBERSUBLIGHT_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES)][subcolor_mask]
         
        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')

        topk = metric_options.get('topk', (1, 5))
        thrs = metric_options.get('thrs')
        average_mode = metric_options.get('average_mode', 'macro')

        if True:
            self.saveflag+=1
            savename1="/disk3/zs1/mmclassification/work_dirs/complexlight_img/onlyhardcase/"+str(self.saveflag)+"hardcase_result.json"
            # savename2="/disk3/zs1/mmclassification/work_dirs/complexlight_img/"+str(self.saveflag+15)+"_result.json"
            # print("==========debug1====================")
            color_scores = scores[:, :len(self.COLOR_CLASSES)]
            pred_color_score = np.max(color_scores, axis=1)
            pred_color_label = np.argmax(color_scores, axis=1)
            pred_color_class = [self.COLOR_CLASSES[lb] for lb in pred_color_label]
            assert color_scores.shape[1] == len(self.COLOR_CLASSES)

            shape_scores = scores[:,
                           len(self.COLOR_CLASSES):len(self.COLOR_CLASSES) + len(
                               self.SHAPE_CLASSES)]
            pred_shape_score = np.max(shape_scores, axis=1)
            pred_shape_label = np.argmax(shape_scores, axis=1)
            pred_shape_class = [self.SHAPE_CLASSES[lb] for lb in pred_shape_label]
            assert shape_scores.shape[1] == len(self.SHAPE_CLASSES)

            toward_scores = scores[:,
                            len(self.COLOR_CLASSES) + len(self.SHAPE_CLASSES):len(
                                self.COLOR_CLASSES) + len(self.SHAPE_CLASSES) + len(
                                self.TOWARD_CLASSES)]
            pred_toward_score = np.max(toward_scores, axis=1)
            pred_toward_label = np.argmax(toward_scores, axis=1)
            pred_toward_class = [self.TOWARD_CLASSES[lb] for lb in pred_toward_label]
            assert toward_scores.shape[1] == len(self.TOWARD_CLASSES)

            character_scores = scores[:,
                               len(self.COLOR_CLASSES) + len(self.SHAPE_CLASSES) + len(
                                   self.TOWARD_CLASSES):len(self.COLOR_CLASSES) + len(self.SHAPE_CLASSES) + len(
                                   self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES)]
            pred_character_score = np.max(character_scores, axis=1)
            pred_character_label = np.argmax(character_scores, axis=1)
            pred_character_class = [self.CHARACTER_CLASSES[lb] for lb in pred_character_label]
            assert character_scores.shape[1] == len(self.CHARACTER_CLASSES)

            simplelight_scores = scores[:, len(self.COLOR_CLASSES) + len(self.SHAPE_CLASSES) + len(
                                   self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES):len(self.COLOR_CLASSES) + len(self.SHAPE_CLASSES) + len(
                                   self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES)+len(self.SIMPLE_CLASSES)]
            pred_simplelight_score = np.max(simplelight_scores, axis=1)
            pred_simplelight_label = np.argmax(simplelight_scores, axis=1)
            pred_simplelight_class = [self.SIMPLE_CLASSES[lb] for lb in pred_simplelight_label]
            assert simplelight_scores.shape[1] == len(self.SIMPLE_CLASSES)
            
            numsublight_scores = scores[:, len(self.COLOR_CLASSES) + len(self.SHAPE_CLASSES) + len(
                                   self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES)+len(self.SIMPLE_CLASSES): len(self.COLOR_CLASSES) + len(self.SHAPE_CLASSES) + len(
                                   self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES)+len(self.SIMPLE_CLASSES)+len(self.NUMBERSUBLIGHT_CLASSES)]
            pred_numsublight_score = np.max(numsublight_scores, axis=1)
            pred_sublight_label = np.argmax(numsublight_scores, axis=1)
            # pred_simplelight_class = [self.SIMPLE_CLASSES[lb] for lb in pred_simplelight_label]
            # assert numsublight_scores.shape[1] ==30
            subcolor_scores = scores[:, len(self.COLOR_CLASSES) + len(self.SHAPE_CLASSES) + len(
                                   self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES)+len(self.SIMPLE_CLASSES)+len(self.NUMBERSUBLIGHT_CLASSES):]
            pred_subcolor_score = np.max(subcolor_scores, axis=1)
            pred_subcolor_label = np.argmax(subcolor_scores, axis=1)
       
            subcolor1_scores=scores[:,len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)+len(self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES)+len(self.SIMPLE_CLASSES)+len(self.NUMBERSUBLIGHT_CLASSES):len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)+len(self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES)+len(self.SIMPLE_CLASSES)+len(self.NUMBERSUBLIGHT_CLASSES)+len(self.COLOR_CLASSES)]
            pred_subcolor1_score = np.max(subcolor1_scores, axis=1)
            pred_subcolor1_label = np.argmax(subcolor1_scores, axis=1)
            subcolor2_scores=scores[:,len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)+len(self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES)+len(self.SIMPLE_CLASSES)+len(self.NUMBERSUBLIGHT_CLASSES)+len(self.COLOR_CLASSES):len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)+len(self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES)+len(self.SIMPLE_CLASSES)+len(self.NUMBERSUBLIGHT_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES)]
            pred_subcolor2_score = np.max(subcolor2_scores, axis=1)
            pred_subcolor2_label = np.argmax(subcolor2_scores, axis=1)
            subcolor3_scores=scores[:,len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)+len(self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES)+len(self.SIMPLE_CLASSES)+len(self.NUMBERSUBLIGHT_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES):len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)+len(self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES)+len(self.SIMPLE_CLASSES)+len(self.NUMBERSUBLIGHT_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES)]
            pred_subcolor3_score = np.max(subcolor3_scores, axis=1)
            pred_subcolor3_label = np.argmax(subcolor3_scores, axis=1)
            subcolor4_scores=scores[:,len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)+len(self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES)+len(self.SIMPLE_CLASSES)+len(self.NUMBERSUBLIGHT_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES):len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)+len(self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES)+len(self.SIMPLE_CLASSES)+len(self.NUMBERSUBLIGHT_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES)]
            pred_subcolor4_score = np.max(subcolor4_scores, axis=1)
            pred_subcolor4_label = np.argmax(subcolor4_scores, axis=1)
            subcolor5_scores=scores[:,len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)+len(self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES)+len(self.SIMPLE_CLASSES)+len(self.NUMBERSUBLIGHT_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES):len(self.COLOR_CLASSES)+len(self.SHAPE_CLASSES)+len(self.TOWARD_CLASSES)+len(self.CHARACTER_CLASSES)+len(self.SIMPLE_CLASSES)+len(self.NUMBERSUBLIGHT_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES)+len(self.COLOR_CLASSES)]
            pred_subcolor5_score = np.max(subcolor5_scores, axis=1)
            pred_subcolor5_label = np.argmax(subcolor5_scores, axis=1)


            img_metas = self.data_infos
           
            outputs1=[]
            outputs2=[]
            count=0
            # print("==========debug2====================")
            for i,img_meta in enumerate(img_metas): 
                result_output = {}
                count=count+1
                ####将结果输出
                result_output['bbox']=img_meta['bbox']
                result_output['simplelight_head'] = img_meta['simplelight_head']
                result_output['toward_head'] = img_meta['toward_head']
                result_output['character_head'] = img_meta['character_head']
                result_output['lightboxshape_head'] = img_meta['lightboxshape_head']
                result_output['lightboxcolor_head'] = img_meta['lightboxcolor_head']
                result_output['sublightcolor_head'] = img_meta["sublightcolor_head"]
                result_output['numsublight_head'] = img_meta["numsublight_head"]
                result_output["imgname"] = img_meta["data_card_id"]+"/"+img_meta["img_info"]["filename"]
                result_output["score_color_pred"] = pred_color_score[i]
                # result_output["score_color_pred"] = pred_color_score
                result_output["label_color_pred"] = pred_color_label[i]
                result_output["label_color_gt"] = img_meta['boxcolor']
                result_output["score_shape_pred"] = pred_shape_score[i]
                # result_output["score_shape_pred"] = pred_shape_score
                result_output["label_shape_pred"] = pred_shape_label[i]
                result_output["label_shape_gt"] = img_meta['boxshape']
                result_output["score_character_pred"] = pred_character_score[i]
                # result_output["score_character_pred"] = pred_character_score
                result_output["label_character_pred"] = pred_character_label[i]
                result_output["label_character_gt"] = img_meta['characteristic']
                result_output["score_toward_pred"] = pred_toward_score[i]
                # result_output["score_toward_pred"] = pred_toward_score
                result_output["label_toward_pred"] = pred_toward_label[i]
                result_output["label_toward_gt"] = img_meta['toward_orientation']
                result_output["score_simplelight_pred"] = pred_simplelight_score[i]
                # result_output["score_simplelight_pred"] = pred_simplelight_score
                result_output["label_simplelight_pred"] = pred_simplelight_label[i]
                result_output["label_simplelight_gt"] = img_meta['simplelight']

                result_output["score_numsublight_pred"] = pred_numsublight_score[i]
                result_output["label_numsublight_pred"] = pred_sublight_label[i]
                result_output["label_numsublight_gt"] = img_meta["numcolorlight"]

                result_output["score_subcolor1_pred"] = pred_subcolor1_score[i]
                result_output["label_subcolor1_pred"] = pred_subcolor1_label[i]
                result_output["label_subcolor1_gt"] = img_meta["subcolor0"]
                result_output["score_subcolor2_pred"] = pred_subcolor2_score[i]
                result_output["label_subcolor2_pred"] = pred_subcolor2_label[i]
                result_output["label_subcolor2_gt"] = img_meta["subcolor1"]
                result_output["score_subcolor3_pred"] = pred_subcolor3_score[i]
                result_output["label_subcolor3_pred"] = pred_subcolor3_label[i]
                result_output["label_subcolor3_gt"] = img_meta["subcolor2"]
                result_output["score_subcolor4_pred"] = pred_subcolor4_score[i]
                result_output["label_subcolor4_pred"] = pred_subcolor4_label[i]
                result_output["label_subcolor4_gt"] = img_meta["subcolor3"]
                result_output["score_subcolor5_pred"] = pred_subcolor5_score[i]
                result_output["label_subcolor5_pred"] = pred_subcolor5_label[i]
                result_output["label_subcolor5_gt"] = img_meta["subcolor4"]


                for key in result_output.keys():
                    result_output[key] = str(result_output[key])
                # if count>1500000:
                outputs1.append(result_output)
                # else:
                #     outputs2.append(result_output)
            # print("==========debug3====================")
            with open(savename1,'w+') as f:
                f.write(json.dumps({"objects":outputs1}, ensure_ascii=False, indent=4))

        if 'accuracy' in metrics:
            if thrs is not None:
                color_acc = accuracy(color_result, color_labels, topk=topk, thrs=thrs)
                shape_acc = accuracy(shape_result, shape_labels, topk=topk, thrs=thrs)
                toward_acc = accuracy(toward_result, toward_labels, topk=topk, thrs=thrs)
                character_acc=accuracy(character_result,character_labels, topk=topk, thrs=thrs)
                simplelight_acc = accuracy(simplelight_result, simplelight_labels, topk=topk, thrs=thrs)
                ##SUBLIGHTS
                numsublight_acc = accuracy(numsublight_result,numsublight_labels, topk=topk,thrs=thrs)

                subcolor1_acc=accuracy(subcolor1_result,subcolor1_labels,topk=topk,thrs=thrs)
                subcolor2_acc=accuracy(subcolor2_result,subcolor2_labels,topk=topk,thrs=thrs)
                subcolor3_acc=accuracy(subcolor3_result,subcolor3_labels,topk=topk,thrs=thrs)
                subcolor4_acc=accuracy(subcolor4_result,subcolor4_labels,topk=topk,thrs=thrs)
                subcolor5_acc=accuracy(subcolor5_result,subcolor5_labels,topk=topk,thrs=thrs)



            else:
                color_acc = accuracy(color_result, color_labels, topk=topk)
                shape_acc = accuracy(shape_result, shape_labels, topk=topk)
                toward_acc = accuracy(toward_result, toward_labels, topk=topk)
                character_acc=accuracy(character_result,character_labels, topk=topk)
                simplelight_acc = accuracy(simplelight_result, simplelight_labels, topk=topk)

                numsublight_acc = accuracy(numsublight_result,numsublight_labels, topk=topk)
                subcolor1_acc=accuracy(subcolor1_result,subcolor1_labels,topk=topk)
                subcolor2_acc=accuracy(subcolor2_result,subcolor2_labels,topk=topk)
                subcolor3_acc=accuracy(subcolor3_result,subcolor3_labels,topk=topk)
                subcolor4_acc=accuracy(subcolor4_result,subcolor4_labels,topk=topk)
                subcolor5_acc=accuracy(subcolor5_result,subcolor5_labels,topk=topk)
            if isinstance(topk, tuple):
                eval_results_ = {
                    f'accuracy_top-{k}': a
                    for k, a in zip(topk, acc)
                }
            else:
                eval_results_ = {'color_accuracy': color_acc, 'shape_accuracy': shape_acc, 'toward_accuracy': toward_acc,'character_accuracy':character_acc,'simplelght_accuracy':simplelight_acc,'numsublight_accuracy':numsublight_acc,"subcolor1_acc":subcolor1_acc,"subcolor2_acc":subcolor2_acc,"subcolor3_acc":subcolor3_acc,"subcolor4_acc":subcolor4_acc,"subcolor5_acc":subcolor5_acc}
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