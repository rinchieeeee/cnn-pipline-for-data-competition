from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip, CenterCrop, 
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout, 
    IAAAdditiveGaussianNoise, Transpose, Blur
    )
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
import random
import numpy as np
import math

class RandomErasing(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        super(RandomErasing, self).__init__(always_apply=always_apply, p=p) # (1) 継承元のクラスへ渡すパラメータ（＝おまじまい）
        self.mean = mean # (2)パラメータ
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def apply(self, img, **params):
        # (3) 実行. 100回チャレンジして, 条件分に入ったら実行されるよ
        for attempt in range(100):
            area = img.shape[0] * img.shape[1]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1) 

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[1] and h < img.shape[0]:
                x1 = random.randint(0, img.shape[0] - h)
                y1 = random.randint(0, img.shape[1] - w)
                if len(img.shape) == 3:
                    img[x1:x1 + h, y1:y1 + w, 0] = self.mean[0]
                    img[x1:x1 + h, y1:y1 + w, 1] = self.mean[1]
                    img[x1:x1 + h, y1:y1 + w, 2] = self.mean[2]
                else:
                    img[x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

class FreqMask(ImageOnlyTransform):
    def __init__(self, always_apply = False, p = 0.5, F = 35, max_mask_num = 4, replace_with_zero = False):
        super(FreqMask, self).__init__(always_apply = always_apply, p = p)
        self.max_mask_num = max_mask_num
        self.F = F
        self.replace_with_zero = replace_with_zero

    def apply(self, spectrum, **params):
        spectrum_len = spectrum.shape[0] # パワースペクトル
        num_mask = np.random.randint(1, self.max_mask_num)
    
        for i in range(0, num_mask):        
            f = random.randrange(10, self.F)
            f_zero = random.randrange(0, spectrum_len - f)

            # avoids randrange error if values are equal and range is empty
            if (f_zero == f_zero + f): 
                return spectrum

            mask_end = random.randrange(f_zero, f_zero + f) 
            if (self.replace_with_zero): 
                spectrum[f_zero:mask_end] = 0
            else: 
                spectrum[f_zero:mask_end] = spectrum.mean()
    
        return spectrum

class TimeMask(ImageOnlyTransform):
    def __init__(self, always_apply = False, p = 0.5, F = 35, max_mask_num = 4, replace_with_zero = False):
        super(TimeMask, self).__init__(always_apply = always_apply, p = p)
        self.max_mask_num = max_mask_num
        self.F = F
        self.replace_with_zero = replace_with_zero

    def apply(self, spectrum, **params):
        time_series_len = spectrum.shape[1]
        num_mask = np.random.randint(1, self.max_mask_num)

        for i in range(0, num_mask):
            f = random.randrange(10, self.F)
            f_zero = random.randrange(0, time_series_len - f)

            if (f_zero == f_zero + f):
                return spectrum

            mask_end = random.randrange(f_zero, f_zero + f)
            if (self.replace_with_zero):
                spectrum[:, f_zero:mask_end] = 0
            else:
                spectrum[:, f_zero:mask_end] = spectrum.mean()
        
        return spectrum 


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