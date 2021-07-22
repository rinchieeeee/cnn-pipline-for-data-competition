from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip, CenterCrop, 
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout, 
    IAAAdditiveGaussianNoise, Transpose, Blur
    )
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform


def get_transforms(config: dict, data_type : str):

    transforms = config["transforms"]
    
    if transforms is None:
        return None
    
    transform_list = []
    if data_type == 'train':
        for aug in transforms["train"]:
            transforms_name = aug["name"]
            transforms_params = {} if aug["params"] is None else aug["params"]
            if globals().get(transforms_name) is not None:
                transforms_func = globals()[transforms_name]  # globals()で, global変数が取得している
                transform_list.append(transforms_func(**transforms_params))
    
        if len(transform_list) > 1:
            return Compose(transform_list)
        else:
            return None

    elif data_type == 'valid':

        for aug in transforms["valid"]:
            transforms_name = aug["name"]
            transforms_params = {} if transforms_name["params"] is None else transforms_name["params"]
            if globals().get(transforms_name) is not None:
                transforms_func = globals()[transforms_name]
                transform_list.append(transforms_func(**transforms_params))

        if len(transform_list) > 1:
            return Compose(transform_list)

        else:
            return None