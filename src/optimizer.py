from torch.optim import Adam, SGD
import yaml

def get_optimizer(learning_params, config: dict):
    """
    learning_params: learnable parameter of model
    config: original dict opning yaml file
    """

    config_optim = config["optimizer"]

    if config_optim["name"] == "Adam":
        return Adam(learning_params, **config_optim["params"])


    if config_optim["name"] == "SGD":
        return SGD()