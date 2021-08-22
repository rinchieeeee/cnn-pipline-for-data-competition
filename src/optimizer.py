from torch.optim import Adam, SGD, AdamW
import yaml

def get_optimizer(learning_params, config: dict):
    """
    learning_params: learnable parameter of model
    config: original dict opning yaml file
    """

    config_optim = config["optimizer"]

    if config_optim["name"] == "Adam":
        return Adam(learning_params, **config_optim["params"])


    if config_optim["name"] == "AdamW":
        return AdamW(learning_params, **config_optim["params"])
        
    if config_optim["name"] == "SGD":
        return SGD(learning_params, **config_optim["params"])



def get_optimizer_for_lr_find(learning_params, lr_init, config: dict):
    """
    learning_params: learnable parameter of model
    config: original dict opning yaml file
    """

    config_optim = config["optimizer"]

    if config_optim["name"] == "Adam":
        return Adam(learning_params, lr = lr_init, **config_optim["params"])


    if config_optim["name"] == "AdamW":
        return AdamW(learning_params, lr = lr_init, **config_optim["params"])

    if config_optim["name"] == "SGD":
        return SGD(learning_params, lr = lr_init, **config_optim["params"])