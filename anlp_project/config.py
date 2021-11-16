from pathlib import Path
from pydoc import locate

from rich.pretty import pprint
from yaml import safe_load


class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    @staticmethod
    def from_file(file_path=Path(__file__).parent / Path("hparams.yaml")):
        """
        Reads configuration from file
        """
        c = Config()
        hparams = safe_load(Path(file_path).read_text())["hparams"]
        for hp in hparams:
            hp_name = hp["name"].replace("-", "_")
            hp_type = locate(hp["type"])

            if "default" in hp:
                hp_value = hp_type(hp["default"])
            else:
                hp_value = None

            c.__setattr__(hp_name, hp_value)

        return c

    def dump(self):
        """
        Prints all properties for this object
        """
        pprint(self.__dict__)
