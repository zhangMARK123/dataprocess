# model settings
model = dict(
    type='MultiHeadClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,
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
        in_channels=256,
        freeze_color_head=False,
        freeze_shape_head=False,
        freeze_toward_head=False,
        freeze_character_head=False,
        freeze_simplelight_head=False,
        # loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
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
dataset_root = '/disk3/zbh/Datasets/'
img_norm_cfg = dict(
    mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='SimDet', prob=0.3),
    dict(type='Rotateimg',prob=0.3),
    # dict(type='Colorshuffle',prob=0.2),
    dict(type='RandomFlip', flip_prob=0.0, direction='horizontal'),
    dict(type='ResizeAndPad', size=128),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor',
         keys=['boxcolor', "lightboxcolor_head", 'boxshape', "lightboxshape_head", 'toward_orientation', "toward_head",
               'characteristic', "character_head", "simplelight", "simplelight_head"]),
    dict(type='Collect',
         keys=['img', 'boxcolor', "lightboxcolor_head", 'boxshape', "lightboxshape_head", 'toward_orientation',
               "toward_head", 'characteristic', "character_head", 'simplelight', 'simplelight_head'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='SimDet', prob=0.),
    dict(type='ResizeAndPad', size=128),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'], meta_keys=('filename', 'ori_filename', 'ori_shape',
                                                  'img_shape', 'flip', 'flip_direction',
                                                  'img_norm_cfg', 'boxcolor', "lightboxcolor_head", 'boxshape',
                                                  "lightboxshape_head",
                                                  'toward_orientation', 'toward_head', 'characteristic',
                                                  "character_head", 'simplelight', 'simplelight_head'))
]
data = dict(
    # sampler=dict(type='GroupSampler'),
    samples_per_gpu=32 * 8,
    workers_per_gpu=32 * 2,
    train=[dict(
        type=dataset_type,
        ann_file='/disk3/zbh/Datasets/2022_train_GW_1.json',
        data_prefix='/disk3/zbh/Datasets/2022_Q1_icu30_crop/',
        pipeline=train_pipeline),
        dict(
        type=dataset_type,
        ann_file='/disk3/zbh/Datasets/2022_train_BB_1.json',
        data_prefix='/disk3/zbh/Datasets/2022_Q1_icu30_crop/',
        pipeline=train_pipeline),
         dict(
        type=dataset_type,
        ann_file='/disk3/zbh/Datasets/2022_Q1_icu30_moni.json',
        data_prefix='/disk3/zbh/Datasets/2022_Q1_icu30_crop/',
        pipeline=train_pipeline)],
        
    val=dict(
        type=dataset_type,
        ann_file=dataset_root + "2022_QA_test_BB.json",
        data_prefix=dataset_root + '2022_Q1_icu30_test_crop/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=dataset_root + "2022_QA_test_BB.json",
        data_prefix=dataset_root + '2022_Q1_icu30_test_crop/',
        pipeline=test_pipeline))
evaluation = dict(interval=2000, metric='accuracy', metric_options={'topk': 1})

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[12000, 16000])

# Runner type
runner = dict(type='IterBasedRunner', max_iters=24000)

# checkpoint saving
checkpoint_config = dict(interval=2000)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = "./work_dirs/ori_model/ALL_addmoni"
