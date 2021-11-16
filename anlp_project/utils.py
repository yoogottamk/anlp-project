from functools import wraps
from pathlib import Path
from pydoc import locate

import click
import yaml


def cli_decorator(f):
    @wraps(f)
    def __cli_decorator(*args, **kwargs):
        _f = click.command("anlp_project")(f)
        _f = click.argument("subcmd")(_f)

        hparams = yaml.safe_load(
            (Path(__file__).parent / Path("hparams.yaml")).read_text()
        )["hparams"]

        for hp in hparams[::-1]:
            hp_type = locate(hp.pop("type"))
            flag_names = ["--" + hp.pop("name")]
            if "short_name" in hp:
                flag_names.append("-" + hp.pop("short_name"))

            if "required" in hp:
                # remove the default value since user will need to specify it anyways
                # and click will throw error otherwise
                hp.pop("default", None)

            if "default" in hp:
                hp["default"] = hp_type(hp["default"])

            # at this point, we have removed all non-standard click stuff
            # we can simply pass the hp dict as-is to click, making it
            # fully configurable by the user

            if hp_type is bool:
                _f = click.option(
                    *flag_names,
                    is_flag=True,
                    **hp,
                )(_f)
            else:
                _f = click.option(
                    *flag_names,
                    type=hp_type,
                    **hp,
                )(_f)

        return _f(*args, **kwargs)

    return __cli_decorator
