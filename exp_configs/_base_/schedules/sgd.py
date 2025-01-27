# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

# optimizer
optimizer = dict(
    type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
