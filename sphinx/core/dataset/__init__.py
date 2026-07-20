from .feature_v1 import FeatureBase


def gen_dataset(**config):
    dataset = eval(config["name"])(**{k: v for k, v in config.items() if k != "name"})
    return dataset
