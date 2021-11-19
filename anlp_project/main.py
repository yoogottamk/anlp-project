from datetime import datetime
import logging

from anlp_project.config import Config
from anlp_project.train import train_model
from anlp_project.inference import inference_model
from anlp_project.utils import cli_decorator


@cli_decorator
def anlp_project(**kwargs):
    config = Config(**kwargs)

    curr_time = datetime.now().isoformat()
    logging.basicConfig(filename=config.logfile.format(curr_time), level=logging.WARNING)

    if not kwargs.get("disable_print_config", False):
        config.dump()

    if kwargs["subcmd"] == "train":
        train_model(config)
    if kwargs["subcmd"] == "inference":
        inference_model(config)


if __name__ == "__main__":
    anlp_project()
