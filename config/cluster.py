import argparse
from utils import yaml_config_hook
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
