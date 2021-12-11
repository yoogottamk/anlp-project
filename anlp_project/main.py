import logging

from rich.logging import RichHandler

from anlp_project.config import Config
from anlp_project.inference import inference_model, inference_t5
from anlp_project.train import train_model
from anlp_project.utils import cli_decorator


@cli_decorator
def anlp_project(**kwargs):
    """
    Entrypoint for `anlp_project` CLI
    """
    config = Config(**kwargs)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()],
    )

    if not kwargs.get("disable_print_config", False):
        config.dump()

    if kwargs["subcmd"] == "train":
        train_model(config)
    if kwargs["subcmd"] == "inference":
        print(inference_t5(kwargs["checkpoint"], kwargs["sentence"]))


if __name__ == "__main__":
    anlp_project()
