from .WideDeep import WideDeepGATProb


def gen_model(**config):
    model = eval(config["name"])(**{k: v for k, v in config.items() if k != "name"})
    return model
