import glob
import os
import pickle
import sys
import json
import cv2

import lmdb
import numpy as np
from tqdm import tqdm



def loadjson(root_dir,labelpath):
    labels=json.loads(open(os.path.join(root_dir,labelpath),'r').read())
    imglist=[]
    for obj in labels["objects"]:
        path=os.path.join(obj["data_card_id"],obj["img_info"]["filename"])
        imglist.append(path)
    return imglist

def generate_from_imagefolder(opt):
    """
    Create lmdb for general image folders
    If all the images have the same resolution, it will only store one copy of resolution info.
        Otherwise, it will store every resolution info.
    """
    img_folder = opt['root_folder']  
    meta_info = {'name': opt['lmdb_save_name']}
    
    labelpath=opt['labelpath']
    sizeimg_path=opt['sizeimg_path']
    lmdb_save_path = os.path.join(img_folder, meta_info['name']+".lmdb")


    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with 'lmdb'.")
    if not os.path.exists(lmdb_save_path):
        print(f"{lmdb_save_path} 不存在，创建...")
        os.makedirs(lmdb_save_path)
    # read all the image paths to a list

    print('Reading image path list ...')
    all_img_list=loadjson(img_folder,labelpath)
    assert len(all_img_list) != 0, "请确认后缀名正确"
    # cache the filename, 这里的文件名必须是ascii字符
    keys = []

    for img_path in all_img_list:
        keys.append(img_path)

    # create lmdb environment

    # 估算大概的映射空间大小
    # 345.5
    data_size_per_img = cv2.imread(os.path.join(img_folder,sizeimg_path),
                                   cv2.IMREAD_UNCHANGED).nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_img_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)
    # map_size：
    # Maximum size database may grow to; used to size the memory mapping. If database grows larger
    # than map_size, an exception will be raised and the user must close and reopen Environment.

    # write data to lmdb

    txn = env.begin(write=True)
    resolutions = []
    tqdm_iter = tqdm(enumerate(zip(all_img_list, keys)),
                     total=len(all_img_list),
                     leave=False)
    for idx, (path, key) in tqdm_iter:
        tqdm_iter.set_description('Write {}'.format(key))

        key_byte = key.encode('ascii')
        img = cv2.imread(os.path.join(img_folder,path))
        data= cv2.imencode('.jpg', img)[1]
        if img.ndim == 2:
            H, W = img.shape
            C = 1
        else:
            H, W, C = img.shape
        resolutions.append('{:d}_{:d}_{:d}'.format(C, H, W))

        txn.put(key_byte, data)
        if (idx + 1) % opt['commit_interval'] == 0:
            txn.commit()
            # commit 之后需要再次 begin
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    print('Finish writing lmdb.')

    # create meta information

    # check whether all the images are the same size
    assert len(keys) == len(resolutions)
    if len(set(resolutions)) <= 1:
        meta_info['resolution'] = [resolutions[0]]
        meta_info['keys'] = keys
        print('All images have the same resolution. Simplify the meta info.')
    else:
        meta_info['resolution'] = resolutions
        meta_info['keys'] = keys
        print(
            'Not all images have the same resolution. Save meta info for each image.'
        )

    pickle.dump(meta_info,
                open(os.path.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')
def test_lmdb(dataroot, index=1):
    env = lmdb.open(dataroot,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False)
    meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'),
                                 "rb"))
    print('Name: ', meta_info['name'])
    print('Resolution: ', meta_info['resolution'])
    print('# keys: ', len(meta_info['keys']))

    # read one image
    key = meta_info['keys'][index]
    print('Reading {} for test.'.format(key))
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    
    
    img = cv2.imdecode(img_flat, cv2.IMREAD_UNCHANGED)
    #cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)

    # C, H, W = [int(s) for s in meta_info['resolution'][index].split('_')]
    # img = img_flat.reshape(H, W, C)

    cv2.namedWindow('Test')
    cv2.imshow('Test', img)
    cv2.waitKeyEx()
if __name__ == "__main__":
    opt = {
            'root_folder':r"C:\Users\zxpan\Desktop\daily_report\data\Q3Q4_chengyongche\crop_img",
            'labelpath': "CYC_WLC_classfic_zs_train.json",
            'lmdb_save_name': "CYC_WLC_classfic_zs_train",
            'sizeimg_path':r"61724a6efb21ccdc3d62b8d5\images\1634115616877081_1.jpg",
            'commit_interval': 100,
            # After commit_interval images, lmdb commits
            'num_workers': 8,
        }
    #generate_from_imagefolder(opt)

    lmdbpath=r"C:\Users\zxpan\Desktop\daily_report\data\Q3Q4_chengyongche\crop_img\CYC_WLC_classfic_zs_train.lmdb"
    test_lmdb(lmdbpath, index=1)
   