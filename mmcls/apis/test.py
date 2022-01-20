# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
import json

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    **show_kwargs):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        batch_size = len(result)
        results.extend(result)

        if show or out_dir:
            scores = np.vstack(result)
            pred_score = np.max(scores, axis=1)
            pred_label = np.argmax(scores, axis=1)
            pred_class = [model.CLASSES[lb] for lb in pred_label]

            img_metas = data['img_metas'].data[0]
            imgs = tensor2imgs(data['img'], **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                result_show = {
                    'pred_score': pred_score[i],
                    'pred_label': pred_label[i],
                    'pred_class': pred_class[i]
                }
                model.module.show_result(
                    img_show,
                    result_show,
                    show=show,
                    out_file=out_file,
                    **show_kwargs)

        batch_size = data['img'].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results
def single_gpu_test_sublights(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    result_dir=None,
                    **show_kwargs):
    model.eval()
    results = []
    outputs=[]
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        batch_size = len(result)
        results.extend(result)
       
        

        if show or out_dir:
            scores = np.vstack(result)
            color_scores = scores[:, :len(dataset.COLOR_CLASSES)]
            pred_color_score = np.max(color_scores, axis=1)
            pred_color_label = np.argmax(color_scores, axis=1)
            pred_color_class = [dataset.COLOR_CLASSES[lb] for lb in pred_color_label]
            assert color_scores.shape[1] == len(dataset.COLOR_CLASSES)

            shape_scores = scores[:, len(data_loader.dataset.COLOR_CLASSES):len(data_loader.dataset.COLOR_CLASSES)+len(data_loader.dataset.SHAPE_CLASSES)]
            pred_shape_score = np.max(shape_scores, axis=1)
            pred_shape_label = np.argmax(shape_scores, axis=1)
            pred_shape_class = [data_loader.dataset.SHAPE_CLASSES[lb] for lb in pred_shape_label]
            assert shape_scores.shape[1] == len(dataset.SHAPE_CLASSES)

            toward_scores = scores[:, len(data_loader.dataset.COLOR_CLASSES)+len(data_loader.dataset.SHAPE_CLASSES):len(data_loader.dataset.COLOR_CLASSES)+len(data_loader.dataset.SHAPE_CLASSES)+len(data_loader.dataset.TOWARD_CLASSES)]
            pred_toward_score = np.max(toward_scores, axis=1)
            pred_toward_label = np.argmax(toward_scores, axis=1)
            pred_toward_class = [dataset.TOWARD_CLASSES[lb] for lb in pred_toward_label]
            assert toward_scores.shape[1] == len(dataset.TOWARD_CLASSES)

            character_scores = scores[:, len(data_loader.dataset.COLOR_CLASSES)+len(data_loader.dataset.SHAPE_CLASSES)+len(data_loader.dataset.TOWARD_CLASSES):-len(data_loader.dataset.SIMPLE_CLASSES)]
            pred_character_score = np.max(character_scores, axis=1)
            pred_character_label = np.argmax(character_scores, axis=1)
            pred_character_class = [dataset.CHARACTER_CLASSES[lb] for lb in pred_character_label]
            assert character_scores.shape[1] == len(dataset.CHARACTER_CLASSES)

            simplelight_scores = scores[:, -len(data_loader.dataset.SIMPLE_CLASSES):]
            pred_simplelight_score = np.max(simplelight_scores, axis=1)
            pred_simplelight_label = np.argmax(simplelight_scores, axis=1)
            pred_simplelight_class = [dataset.SIMPLE_CLASSES[lb] for lb in pred_simplelight_label]
            assert simplelight_scores.shape[1] == len(dataset.SIMPLE_CLASSES)

            img_metas = data['img_metas'].data[0]
            imgs = tensor2imgs(data['img'], **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                result_show = {}
                result_output={}
                SHOW_FLAG = False
                
                if img_meta['lightboxcolor_head']:
                    # result_show["score_color_pred"] = pred_color_score[i]
                    result_show["label_color_pred"] = pred_color_label[i]
                    result_show["label_color_gt"] = img_meta['boxcolor']
                    if result_show["label_color_pred"] != result_show["label_color_gt"]:
                        SHOW_FLAG = True
                if img_meta['lightboxshape_head']:
                    # result_show["score_shape_pred"] = pred_shape_score[i]
                    result_show["label_shape_pred"] = pred_shape_label[i]
                    result_show["label_shape_gt"] = img_meta['boxshape']
                    if result_show["label_shape_pred"] != result_show["label_shape_gt"]:
                        SHOW_FLAG = True
                
                if img_meta['character_head']:
                    # result_show["score_character_pred"] = pred_character_score[i]
                    result_show["label_character_pred"] = pred_character_label[i]
                    result_show["label_character_gt"] = img_meta['characteristic']
                    if result_show["label_character_pred"] != result_show["label_character_gt"]:
                        SHOW_FLAG = True
                if img_meta['toward_head']:
                    # result_show["score_toward_pred"] = pred_toward_score[i]
                    result_show["label_toward_pred"] = pred_toward_label[i]
                    result_show["label_toward_gt"] = img_meta['toward_orientation']
                    if result_show["label_toward_pred"] != result_show["label_toward_gt"]:
                    #if result_show["label_toward_pred"] == result_show["label_toward_gt"] == 0:
                        SHOW_FLAG = True
                if img_meta['simplelight_head']:
                    # result_show["score_simplelight_pred"] = pred_simplelight_score[i]
                    result_show["label_simplelight_pred"] = pred_simplelight_label[i]
                    result_show["label_simplelight_gt"] = img_meta['simplelight']
                    if result_show["label_simplelight_pred"] != result_show["label_simplelight_gt"]:
                    #if result_show["label_toward_pred"] == result_show["label_toward_gt"] == 0:
                        SHOW_FLAG = True
                ####将结果输出
                result_output['simplelight_head']=img_meta['simplelight_head']
                result_output['toward_head']=img_meta['toward_head']
                result_output['character_head']=img_meta['character_head']
                result_output['lightboxshape_head']=img_meta['lightboxshape_head']
                result_output['lightboxcolor_head']=img_meta['lightboxcolor_head']
                result_output["imgname"]=img_meta['ori_filename']
                result_output["score_color_pred"] = pred_color_score[i]
                result_output["label_color_pred"] = pred_color_label[i]
                result_output["label_color_gt"] = img_meta['boxcolor']             
                result_output["score_shape_pred"] = pred_shape_score[i]
                result_output["label_shape_pred"] = pred_shape_label[i]
                result_output["label_shape_gt"] = img_meta['boxshape']    
                result_output["score_character_pred"] = pred_character_score[i]
                result_output["label_character_pred"] = pred_character_label[i]
                result_output["label_character_gt"] = img_meta['characteristic']    
                result_output["score_toward_pred"] = pred_toward_score[i]
                result_output["label_toward_pred"] = pred_toward_label[i]
                result_output["label_toward_gt"] = img_meta['toward_orientation']         
                result_output["score_simplelight_pred"] = pred_simplelight_score[i]
                result_output["label_simplelight_pred"] = pred_simplelight_label[i]
                result_output["label_simplelight_gt"] = img_meta['simplelight']                   
                for key in result_output.keys():
                    result_output[key]=str(result_output[key])
                outputs.append(result_output)
                if SHOW_FLAG:   # and img_meta['toward_head']:
                    model.module.show_result(
                        img,
                        result_show,
                        font_scale=0.3,
                        show=show,
                        out_file=out_file,
                        **show_kwargs)

        batch_size = data['img'].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    

    if result_dir:     
        with open(result_dir, 'w+') as f:
            json.dump({'objects':outputs},f)
    return results

def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        # Check if tmpdir is valid for cpu_collect
        if (not gpu_collect) and (tmpdir is not None and osp.exists(tmpdir)):
            raise OSError((f'The tmpdir {tmpdir} already exists.',
                           ' Since tmpdir will be deleted after testing,',
                           ' please make sure you specify an empty one.'))
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)

        if rank == 0:
            batch_size = data['img'].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_result = mmcv.load(part_file)
            part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_result = pickle.loads(recv[:shape[0]].cpu().numpy().tobytes())
            part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
