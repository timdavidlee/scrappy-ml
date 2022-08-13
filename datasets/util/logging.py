import logging
import logging.config
import yaml


def get_logger():
    with open("datasets/util/logging.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    logging.config.dictConfig(config)

    return logging.getLogger("scrappy_ml")
