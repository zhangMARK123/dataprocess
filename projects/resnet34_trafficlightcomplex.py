# model settings
model = dict(
    type='MultiHeadClassifier',
    backbone=dict(
        type='ResNet',
        depth=34,
        num_stages=4,
        out_indices=(3,),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='TrafficSubLightcomplexMultiClsHead',
        shape_classes=8,
        color_classes=6,
        toward_classes=4,
        character_classes=6,
        simplelight_classes=2,
        in_channels=512,
        freeze_color_head=False,
        freeze_shape_head=False,
        freeze_toward_head=False,
        freeze_character_head=False,
        freeze_simplelight_head=False,
        loss=dict(
            type='FocalLoss',
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        topk=1,
        cal_acc=True)
)
# dataset settings
dataset_type = 'TrafficSubLightcomplexClsDataset'
dataset_root = '/share/zbh/Datasets/'
img_norm_cfg = dict(
    mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CropByBbox', save_crop_result=False),
    dict(type='RandomFlip', flip_prob=0.0, direction='horizontal'),
    dict(type='ResizeAndPad', size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor',
         keys=['boxcolor', "lightboxcolor_head", 'boxshape', "lightboxshape_head", 'toward_orientation', "toward_head",
               'characteristic', "character_head","simplelight","simplelight_head"]),
    dict(type='Collect',
         keys=['img', 'boxcolor', "lightboxcolor_head", 'boxshape', "lightboxshape_head", 'toward_orientation',
               "toward_head", 'characteristic', "character_head",'simplelight','simplelight_head'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),

    dict(type='CropByBbox', save_crop_result=False),
    dict(type='ResizeAndPad', size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'], meta_keys=('filename', 'ori_filename', 'ori_shape',
                                                  'img_shape', 'flip', 'flip_direction',
                                                  'img_norm_cfg', 'boxcolor', "lightboxcolor_head", 'boxshape',
                                                  "lightboxshape_head",
                                                  'toward_orientation', 'toward_head', 'characteristic',
                                                  "character_head",'simplelight','simplelight_head'))
]
data = dict(
    #sampler=dict(type='ImbalancedDatasetSampler',dataset=dataset),
    samples_per_gpu=32 * 1,
    workers_per_gpu=32,
    train=dict(
        type=dataset_type,
        ann_file=dataset_root + "2022_Q1_icu30_new_crop_correct4_zs_train.json",
        # ann_file=dataset_root + "traffic_light_test.json",
        # ann_file=dataset_root + "traffic_light_debug.json",
        data_prefix=dataset_root+'2022_Q1_icu30_new_crop2_correct/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=dataset_root + "2022_Q1_icu30_new_crop_correct4_zs_test.json",
        # ann_file=dataset_root + "traffic_light_debug.json",
        data_prefix=dataset_root+'2022_Q1_icu30_new_crop2_correct/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=dataset_root + "2022_Q1_icu30_new_crop_correct4_zs_test.json",
        data_prefix=dataset_root+'2022_Q1_icu30_new_crop2_correct/' ,
        pipeline=test_pipeline))
evaluation = dict(interval=10, metric='accuracy', metric_options={'topk': 1})

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[30, 60, 90])
runner = dict(type='EpochBasedRunner', max_epochs=100)

# checkpoint saving
checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook')
        #dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = "./work_dirs/resnet34_trafficlightcomplex19/"
