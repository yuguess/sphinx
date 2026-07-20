from .WideDeep import WideDeep_v1

def gen_model(**config):
    model = eval(config["name"])(**{k: v for k, v in config.items() if k != "name"})
    return model
