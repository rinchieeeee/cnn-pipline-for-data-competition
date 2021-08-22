import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import yaml
from arcface import ArcFaceLayer

def adaptive_concat_avgmax_pool2d(x, output_size = 1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return torch.cat((x_avg, x_max), dim = 1)

class AdaptiveCatAvgMaxPool2d(nn.Module):
    def __init__(self, output_size = 1):
        super(AdaptiveCatAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_concat_avgmax_pool2d(x, self.output_size)

class SelectAdaptivePool2d(nn.Module):
    """
    Selectable global pooling layer with dynamic input kernel size
    """
    def __init__(self, output_size = 1, pool_type = 'avg', flatten = True):
        super(SelectAdaptivePool2d, self).__init__()
        
        self.pool_type = pool_type or ''  # convert other falsy values to empty string for consistent TS typing
        self.flatten = nn.Flatten(1) if flatten else nn.Identity()
        
        if pool_type == '':
            self.pool = nn.Identity()  # pass through
        elif pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(output_size)
        elif pool_type == 'concat_avg_max':
            self.pool = AdaptiveCatAvgMaxPool2d(output_size)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        else:
            assert False, 'Invalid pool type: %s' % pool_type

    def is_identity(self):
        return not self.pool_type

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)



class CustomNFNet(nn.Module):

    def __init__(self, model_name, class_num, in_channels, pretrained = False):
        super().__init__()
        self.model = timm.create_model(model_name, in_chans = in_channels, pretrained = pretrained)
        n_features = self.model.head.fc.in_features
        self.model.head.fc = nn.Linear(n_features, class_num)

    def forward(self, x):
        x = self.model(x)
        return x

class CustomRexNet_Base(nn.Module):

    def __init__(self, model_name, class_num, in_channels, 
                    dropout_rate = 0.0, global_pool_type = "avg", pretrained = False):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.global_pool_type = global_pool_type
        self.model = timm.create_model(model_name, in_chans = in_channels, pretrained = pretrained)
        n_features = self.model.head.fc.in_features # get the number of unit size in last fully conection layer
        #self.model.fc = nn.Linear(n_features, class_num)

        if self.global_pool_type == "concat_avg_max":
            n_features = n_features * 2
            self.global_pool = SelectAdaptivePool2d(pool_type = self.global_pool_type)
        else:
            self.global_pool = SelectAdaptivePool2d(pool_type = self.global_pool_type)

        self.classifier = nn.Linear(n_features, class_num, bias = True)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.global_pool(x)
        if self.dropout_rate > 0.0:
            x = F.dropout(x, p = self.dropout_rate)

        return self.classifier(x)

class CustomDensNet_Base(nn.Module):

    def __init__(self, model_name, class_num, in_channels, pretrained = False):
        super().__init__()
        self.model = timm.create_model(model_name, in_chans = in_channels, pretrained = pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, class_num)

    def forward(self, x):
        x = self.model(x)
        return x

class CustomResNet_Base(nn.Module):

    def __init__(self, model_name, class_num, in_channels, 
                    dropout_rate = 0.0, global_pool_type = "avg", pretrained = False):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.global_pool_type = global_pool_type
        self.model = timm.create_model(model_name, in_chans = in_channels, pretrained = pretrained)
        n_features = self.model.fc.in_features # get the number of unit size in last fully conection layer
        self.model.fc = nn.Linear(n_features, class_num)

        if self.global_pool_type == "concat_avg_max":
            n_features = n_features * 2
            self.global_pool = SelectAdaptivePool2d(pool_type = self.global_pool_type)
        else:
            self.global_pool = SelectAdaptivePool2d(pool_type = self.global_pool_type)

        self.classifier = nn.Linear(n_features, class_num, bias = True)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.global_pool(x)
        if self.dropout_rate > 0.0:
            x = F.dropout(x, p = self.dropout_rate)

        return self.classifier(x)

class CustomECAResNet_Base(nn.Module):

    def __init__(self, model_name, class_num, in_channels, 
                    dropout_rate = 0.0, global_pool_type = "avg", pretrained = False):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.global_pool_type = global_pool_type
        self.model = timm.create_model(model_name, in_chans = in_channels, pretrained = pretrained)
        n_features = self.model.fc.in_features # get the number of unit size in last fully conection layer
        #self.model.fc = nn.Linear(n_features, class_num)

        if self.global_pool_type == "concat_avg_max":
            n_features = n_features * 2
            self.global_pool = SelectAdaptivePool2d(pool_type = self.global_pool_type)
        else:
            self.global_pool = SelectAdaptivePool2d(pool_type = self.global_pool_type)

        self.classifier = nn.Linear(n_features, class_num, bias = True)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.global_pool(x)
        if self.dropout_rate > 0.0:
            x = F.dropout(x, p = self.dropout_rate)

        return self.classifier(x)

class ResNetArcFace_Base(nn.Module):

    def __init__(self, model_name, class_num, in_channels, 
                    dropout_rate = 0.0, global_pool_type = "avg", pretrained = False):
        super().__init__()
        self.model = timm.create_model(model_name, in_chans = in_channels, pretrained = pretrained)
        n_features = self.model.fc.in_features # get the number of unit size in last fully conection layer
        self.global_pool_type = global_pool_type
        self.dropout_rate = dropout_rate

        if self.global_pool_type == "concat_avg_max":
            n_features = n_features * 2
            self.global_pool = SelectAdaptivePool2d(pool_type = self.global_pool_type)
        else:
            self.global_pool = SelectAdaptivePool2d(pool_type = self.global_pool_type)
        
        self.arcface_layer = ArcFaceLayer(in_features = n_features, class_num = class_num)


    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.global_pool(x)
        if self.dropout_rate > 0.0:
            x = F.dropout(x, p = self.dropout_rate)

        return self.arcface_layer(x)

class CustomTResNet_Base(nn.Module):

    def __init__(self, model_name, class_num, in_channels, pretrained = False, 
                    dropout_rate = 0.0, global_pool_type = "avg"):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.global_pool_type = global_pool_type
        self.model = timm.create_model(model_name, in_chans = in_channels, pretrained = pretrained) 
        n_features = self.model.num_features # get the number of unit size in last fully conection layer
        
        if self.global_pool_type == "concat_avg_max":
            n_features = n_features * 2
            self.global_pool = SelectAdaptivePool2d(pool_type = self.global_pool_type)
        else:
            self.global_pool = SelectAdaptivePool2d(pool_type = self.global_pool_type)
        
        self.classifier = nn.Linear(n_features, class_num, bias = True)


    def forward(self, x):
        x = self.model.forward_features(x) # encoder部分
        x = self.global_pool(x)
        if self.dropout_rate > 0.0:
            x = F.dropout(x, p = self.dropout_rate)
        return self.classifier(x)

class CustomEfficientNet_base(nn.Module):

    def __init__(self, model_name, class_num, in_channels, pretrained = False, 
                    dropout_rate = 0.0, global_pool_type = "avg"):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.global_pool_type = global_pool_type
        self.model = timm.create_model(model_name, in_chans = in_channels, pretrained = pretrained) 
        n_features = self.model.classifier.in_features # get the number of unit size in last fully conection layer
        
        if self.global_pool_type == "concat_avg_max":
            n_features = n_features * 2
            self.global_pool = SelectAdaptivePool2d(pool_type = self.global_pool_type)
        else:
            self.global_pool = SelectAdaptivePool2d(pool_type = self.global_pool_type)
        
        self.classifier = nn.Linear(n_features, class_num, bias = True)


    def forward(self, x):
        x = self.model.forward_features(x) # encoder部分
        x = self.global_pool(x)
        if self.dropout_rate > 0.0:
            x = F.dropout(x, p = self.dropout_rate)
        return self.classifier(x)

class EfficientNetWithArcface_base(nn.Module):

    def __init__(self, model_name, class_num, in_channels, pretrained = False, 
                    dropout_rate = 0.0, global_pool_type = "avg"):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.global_pool_type = global_pool_type
        self.model = timm.create_model(model_name, in_chans = in_channels, pretrained = pretrained) 
        n_features = self.model.classifier.in_features # get the number of unit size in last fully conection layer
        if self.global_pool_type == "concat_avg_max":
            n_features = n_features * 2
            self.global_pool = SelectAdaptivePool2d(pool_type = self.global_pool_type)
        else:
            self.global_pool = SelectAdaptivePool2d(pool_type = self.global_pool_type)
        
        self.arcface_layer = ArcFaceLayer(in_features = n_features, class_num = class_num)


    def forward(self, x):
        x = self.model.forward_features(x) # encoder部分
        x = self.global_pool(x)
        if self.dropout_rate > 0.0:
            x = F.dropout(x, p = self.dropout_rate)
        return self.arcface_layer(x)


class CustomEfficientNetV2_m(nn.Module):

    def __init__(self, model_name, class_num, in_channels, pretrained = False):
        super().__init__()
        self.model = timm.create_model(model_name, in_chans = in_channels, pretrained = pretrained)
        n_features = self.model.classifier.in_features # get the number of unit size in last fully conection layer
        self.model.classifier = nn.Linear(n_features, class_num)


    def forward(self, x):
        x = self.model(x)
        return x

def get_model(config):
    """
    config : original config loading yaml file
    """

    config_model = config["model"]

    if config_model["name"] == "resnet50":
        model = CustomResNet50(model_name = config_model["name"], 
                                class_num = config_model["class_num"], 
                                in_channels = config_model["in_channels"],
                                pretrained = config_model["pretrained"]
                                )

    if config_model["name"] == "resnet34d":
        model = CustomResNet_Base(model_name = config_model["name"], 
                                class_num = config_model["class_num"], 
                                in_channels = config_model["in_channels"],
                                pretrained = config_model["pretrained"],
                                dropout_rate = config_model["pretrained"],
                                global_pool_type = config_model["global_pool_type"]
                                )

    elif config_model["name"] == "ecaresnet50d":
        model = CustomECAResNet_Base(model_name = config_model["name"], 
                                class_num = config_model["class_num"], 
                                in_channels = config_model["in_channels"],
                                pretrained = config_model["pretrained"],
                                dropout_rate = config_model["pretrained"],
                                global_pool_type = config_model["global_pool_type"]
                                )

    if config_model["name"] == "resnet50d":
        model = CustomResNet50(model_name = config_model["name"], 
                                class_num = config_model["class_num"], 
                                in_channels = config_model["in_channels"],
                                pretrained = config_model["pretrained"]
                                )

    elif config_model["name"] == "resnetrs50":
        model = CustomResNet_Base(model_name = config_model["name"], 
                                class_num = config_model["class_num"], 
                                in_channels = config_model["in_channels"],
                                pretrained = config_model["pretrained"]
                                )

    elif config_model["name"] == "resnetrs101":
        model = CustomResNet_Base(model_name = config_model["name"], 
                                class_num = config_model["class_num"], 
                                in_channels = config_model["in_channels"],
                                pretrained = config_model["pretrained"]
                                )
    
    elif config_model["name"] == "resnetrs152":
        model = CustomResNet_Base(model_name = config_model["name"], 
                                class_num = config_model["class_num"], 
                                in_channels = config_model["in_channels"],
                                pretrained = config_model["pretrained"]
                                )

    elif config_model["name"] == "resnet18":
        model = CustomResNet18(model_name = config_model["name"], 
                                class_num = config_model["class_num"], 
                                in_channels = config_model["in_channels"],
                                pretrained = config_model["pretrained"]
                                )

    elif config_model["name"] == "seresnet101":
        model = CustomResNet18(model_name = config_model["name"], 
                                class_num = config_model["class_num"], 
                                in_channels = config_model["in_channels"],
                                pretrained = config_model["pretrained"]
                                )
    elif config_model["name"] == "arcface_resnet":
        model = ResNetArcFace_Base(model_name = config_model["name"][8:], 
                                class_num = config_model["class_num"], 
                                in_channels = config_model["in_channels"],
                                pretrained = config_model["pretrained"],
                                dropout_rate = config_model["pretrained"],
                                global_pool_type = config_model["global_pool_type"]
                                )

    elif config_model["name"] == "resnest":
        model = CustomResNet_Base(model_name = config_model["name"], 
                                class_num = config_model["class_num"], 
                                in_channels = config_model["in_channels"],
                                pretrained = config_model["pretrained"]
                                )
                                
    elif config_model["name"] == "efficientnet_b4":
        model = CustomEfficientNet_base(model_name = config_model["name"], 
                                class_num = config_model["class_num"], 
                                in_channels = config_model["in_channels"],
                                pretrained = config_model["pretrained"]
                                )

    elif config_model["name"] == "efficientnet_b3":
        model = CustomEfficientNet_base(model_name = config_model["name"], 
                                class_num = config_model["class_num"], 
                                in_channels = config_model["in_channels"],
                                pretrained = config_model["pretrained"]
                                )

    elif config_model["name"] == "efficientnet_b5":
        model = CustomEfficientNet_base(model_name = config_model["name"], 
                                class_num = config_model["class_num"], 
                                in_channels = config_model["in_channels"],
                                pretrained = config_model["pretrained"],
                                dropout_rate = config_model["dropout_rate"],
                                global_pool_type = config_model["global_pool_type"]
                                )

    elif config_model["name"] == "efficientnet_b6":
        model = CustomEfficientNet_base(model_name = config_model["name"], 
                                class_num = config_model["class_num"], 
                                in_channels = config_model["in_channels"],
                                pretrained = config_model["pretrained"],
                                dropout_rate = config_model["dropout_rate"],
                                global_pool_type = config_model["global_pool_type"]
                                )
    elif config_model["name"] == "efficientnet_b7":
        model = CustomEfficientNet_base(model_name = config_model["name"], 
                                class_num = config_model["class_num"], 
                                in_channels = config_model["in_channels"],
                                pretrained = config_model["pretrained"],
                                dropout_rate = config_model["dropout_rate"],
                                global_pool_type = config_model["global_pool_type"]
                                )

    elif config_model["name"] == "tf_efficientnet_b3_ns":
        model = CustomEfficientNet_base(model_name = config_model["name"], 
                                class_num = config_model["class_num"], 
                                in_channels = config_model["in_channels"],
                                pretrained = config_model["pretrained"]
                                )

    elif config_model["name"] == "efficientnet_b1":
        model = CustomEfficientNet_base(model_name = config_model["name"], 
                                class_num = config_model["class_num"], 
                                in_channels = config_model["in_channels"],
                                pretrained = config_model["pretrained"],
                                dropout_rate = config_model["dropout_rate"],
                                global_pool_type = config_model["global_pool_type"]
                                )

    elif config_model["name"] == "efficientnet_b2":
        model = CustomEfficientNet_base(model_name = config_model["name"], 
                                class_num = config_model["class_num"], 
                                in_channels = config_model["in_channels"],
                                pretrained = config_model["pretrained"],
                                dropout_rate = config_model["dropout_rate"],
                                global_pool_type = config_model["global_pool_type"]
                                )

    elif config_model["name"] == "tf_efficientnet_b2_ns":
        model = CustomEfficientNet_base(model_name = config_model["name"], 
                                class_num = config_model["class_num"], 
                                in_channels = config_model["in_channels"],
                                pretrained = config_model["pretrained"],
                                dropout_rate = config_model["dropout_rate"],
                                global_pool_type = config_model["global_pool_type"]
                                )

    elif "arcface_efficientnet" in config_model["name"]:
        model = EfficientNetWithArcface_base(model_name = config_model["name"][8:], 
                                class_num = config_model["class_num"], 
                                in_channels = config_model["in_channels"],
                                pretrained = config_model["pretrained"],
                                dropout_rate = config_model["dropout_rate"],
                                global_pool_type = config_model["global_pool_type"]
                                )

    elif config_model["name"] == "efficientnet_b0":
        model = CustomEfficientNet_base(model_name = config_model["name"], 
                                class_num = config_model["class_num"], 
                                in_channels = config_model["in_channels"],
                                pretrained = config_model["pretrained"],
                                dropout_rate = config_model["dropout_rate"],
                                global_pool_type = config_model["global_pool_type"]
                                )


    elif config_model["name"] == "efficientnetv2_m":
        model = CustomEfficientNet_base(model_name = config_model["name"], 
                                class_num = config_model["class_num"], 
                                in_channels = config_model["in_channels"],
                                pretrained = config_model["pretrained"]
                                )

    elif config_model["name"] == "efficientnetv2_s":
        model = CustomEfficientNet_base(model_name = config_model["name"], 
                                class_num = config_model["class_num"], 
                                in_channels = config_model["in_channels"],
                                pretrained = config_model["pretrained"]
                                )

    elif config_model["name"] == "efficientnetv2_rw_s":
        model = CustomEfficientNet_base(model_name = config_model["name"], 
                                class_num = config_model["class_num"], 
                                in_channels = config_model["in_channels"],
                                pretrained = config_model["pretrained"]
                                )

    elif config_model["name"] == "nfnet_f0":
        model = CustomNFNet(model_name = config_model["name"], 
                            class_num = config_model["class_num"],
                            in_channels = config_model["in_channels"],
                            pretrained = config_model["pretrained"])

    elif "densenet" in config_model["name"]:
        model = CustomDensNet_Base(model_name = config_model["name"], 
                            class_num = config_model["class_num"],
                            in_channels = config_model["in_channels"],
                            pretrained = config_model["pretrained"])

    elif config_model["name"] == "tresnet_m_448":
        model = CustomTResNet_Base(model_name = config_model["name"],
                                    class_num = config_model["class_num"],
                                    in_channels = config_model["in_channels"],
                                    pretrained = config_model["pretrained"],
                                    dropout_rate = config_model["dropout_rate"],
                                    global_pool_type = config_model["global_pool_type"])

    elif "resnest50" in config_model["name"]:
        model = CustomResNet_Base(model_name = config_model["name"],
                                class_num = config_model["class_num"],
                                in_channels = config_model["in_channels"],
                                pretrained = config_model["pretrained"])

    elif "rexnet" in config_model["name"]:
        model = CustomRexNet_Base(model_name = config_model["name"],
                                class_num = config_model["class_num"],
                                in_channels = config_model["in_channels"],
                                pretrained = config_model["pretrained"],
                                dropout_rate = config_model["dropout_rate"],
                                global_pool_type = config_model["global_pool_type"])


    return model