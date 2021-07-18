import torch
import torch.nn as nn
import timm
import yaml


class CustomResNet50(nn.Module):

    def __init__(self, model_name, class_num, pretrained = False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained = pretrained)
        n_features = self.model.fc.in_features # get the number of unit size in last fully conection layer
        self.model.fc = nn.Linear(n_features, class_num)


    def forward(self, x):
        x = self.model(x)
        return x



def get_model(config):
    """
    config : original config loading yaml file
    """

    config_model = config["model"]

    model = CustomResNet50(model_name = config_model["model_name"], 
                            class_num = config_model["class_num"], 
                            pretrained = config_model["pretrained"]
                            )

    return model