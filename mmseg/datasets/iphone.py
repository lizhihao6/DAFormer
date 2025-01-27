# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class IphoneDataset(CustomDataset):
    """Iphone dataset.
    """

    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'plant', 'sky', 'person',
               'rider', 'car', 'truck', 'bus', 'tricycle', 'motorcycle',
               'bicycle')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [70, 130, 180], [220, 20, 60], [255, 0, 0],
               [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230],
               [119, 11, 32]]

    def __init__(self, **kwargs):
        kwargs['classes'] = ['road', 'building', 'fence', 'traffic light',
                             'traffic sign', 'plant', 'sky', 'person', 'car',
                             'truck', 'bus', 'bicycle']
        super(IphoneDataset, self).__init__(**kwargs)
