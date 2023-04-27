custom_imports = dict(
    imports=['mmdet3d.models.voxel_encoders.br_voxel_encoder'],
    allow_failed_imports=False)
custom_imports = dict(
    imports=['mmdet3d.models.backbones.br_second'],
    allow_failed_imports=False)

custom_imports = dict(
    imports=['mmdet3d.models.necks.br_second_fpn'],
    allow_failed_imports=False)

custom_imports=dict(
    imports=['mmdet3d.models.losses.my_loss'])


model = dict(
    voxel_encoder=dict(
        type='BR_HardVFE',
        arg1=xxx,
        arg2=xxx
    ),
    backbone=dict(
        type='BR_SECOND',
        arg1=xxx,
        arg2=xxx
    ),
    neck=dict(
        type='BR_SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
)