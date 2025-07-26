_base_ = [
    '../train_lvis/yolo_world_v2_x_train.py',
]

del _base_.batch_shapes_cfg
del _base_.val_dataloader
del _base_.test_dataloader
del _base_.val_evaluator
del _base_.test_evaluator
del _base_.model_test_cfg
del _base_.optim_wrapper


detector = _base_.model
detector.pop('data_preprocessor')
detector['init_cfg'] = dict(
    type='Pretrained',
    checkpoint=  # noqa: E251
    'pretrained_models/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth'

)
# Add PCS params to YOLOWorldHead
detector['bbox_head']['use_PCS'] = True
detector['bbox_head']['use_aci'] = False
detector['bbox_head']['global_threshold'] = 0.3
detector['bbox_head']['semi_threshold'] = 0.4
detector['bbox_head']['detail_threshold'] = 0.5
del _base_.model

model = dict(
    type='YOLOWorldSort',
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    #data_preprocessor=dict(
        #type='mmdet.TrackDataPreprocessor',
        #mean=[0., 0., 0.],
        #std=[255., 255., 255.],
        #bgr_to_rgb=True,
        #pad_size_divisor=32),
    detector=detector,
    visual=False,
    tracker=dict(
        type='OVTracker',
        init_score_thr=0.0001,
        obj_score_thr=0.0001,
        match_score_thr=0.3,
        memo_frames=10,
        momentum_embed=0.8,
        momentum_obj_score=0.5,
        #match_metric='cosine',
        match_metric='cosine',
        match_with_cosine=True,
        #use_embed='cls_embeds',
        #use_embed='cls_embeds',
        contrastive_thr=0.5,
    )
)

# masa tao
# tracker = dict(
#     type='MasaTaoTracker',
#     init_score_thr=0.0001,
#     obj_score_thr=0.0001,
#     match_score_thr=0.5,
#     memo_tracklet_frames=10,
#     memo_momentum=0.8,
#     with_cats=False,
#     max_distance=-1,
#     fps=1,
# )

# to do: 1. add val dataloader 2. evaluator
optim_wrapper = None

val_cfg = dict(type='ValLoop')

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    visualization=dict(type='mmdet.TrackVisualizationHook', draw=False))


test_pipeline = [
    dict(
        type='TransformBroadcaster',
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='mmyolo.YOLOv5KeepRatioResize', scale=(1333, 800)),
            dict(
                type='mmyolo.LetterResize',
                scale=(1333, 800),
                allow_scale_up=False,
                pad_val=dict(img=114)),
        ]),
    dict(type='mmdet.PackTrackInputs',
         meta_keys=('img_id', 'img_path', 'frame_id', 'ori_shape', 'img_shape', 'scale_factor', 'pad_param', 'texts')
         )
]


# dataloader

test_dataset_tpye = 'Taov1Dataset'
#test_dataset_tpye = 'OVTBDataset'

# check下，加载过程是只加载test数据吗？
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    # Now we support two ways to test, image_based and video_based
    # if you want to use video_based sampling, you can use as follows
    sampler=dict(type='mmdet.TrackImgSampler'),  # image-based sampling
    dataset=dict(
        type=test_dataset_tpye,
        #ann_file='D:/PHDresarch/phd-project/dataset/tao/annotations/tao_val_lvis_v1_classes.json',
        #ann_file='data/OVT-B/ovtb_ann_part.json',
        ann_file='data/tao/annotations/tao_val_lvis_v1_classes.json',
		#ann_file='data/tao/annotations/tao_val_lvis_v1_classes_3x.json',
        data_prefix=dict(img_path='data/tao/frames/'),
        #data_prefix=dict(img_path='data/OVT-B/OVT-B/'),
        #data_prefix=dict(img_path='D:/PHDresarch/phd-project/dataset/tao/frames'),
        test_mode=True,
        pipeline=test_pipeline
    ),
	collate_fn=dict(type='doubleF_collate_fn')
	)

test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(
    type='mmdet.TaoTETAMetric',
    dataset_type=test_dataset_tpye,
    format_only=False,
    #ann_file='D:/PHDresarch/phd-project/dataset/tao/annotations/tao_val_lvis_v1_classes.json',
    #ann_file='data/OVT-B/ovtb_ann_part.json',
	ann_file='data/tao/annotations/tao_val_lvis_v1_classes.json',
    #ann_file='/cfs02/CV/datasets/tao/annotations/tao_val_lvis_v1_classes.json', # in a100
    metric=['TETA'],
    outfile_prefix='results/yolow_track_results/yolow-track-clembed-ovmot-test',
    open_vocabulary=True,
    #remap_file = 'data/tao/annotations/image_id_map_3x_part.json',
)

test_evaluator = val_evaluator

model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=False,
    # The number of boxes before NMS
    nms_pre=30000,
    score_thr=0.1,  # Threshold to filter out boxes.
    nms=dict(type='nms', iou_threshold=0.7),  # NMS type and threshold
    max_per_img=300)  # Max number of detections of each image



# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'embedding': dict(lr_mult=1.0),
            'norm': dict(decay_mult=0.0),
        },
        base_total_batch_size=8  # 确保这里设置了 base_total_batch_size
    ),
    clip_grad=dict(max_norm=35, norm_type=2))

# 学习率调度器
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# 覆盖默认钩子配置
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='TaoCheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='tao_teta_metric/TAO',
        rule='greater',
        detector_ckpt_path='pretrained_models/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.TrackVisualizationHook', draw=False))

# 自动恢复训练
resume = False