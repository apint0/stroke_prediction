"""Set of tools to define and load configurations."""

import importlib
import os
import sys


def set_import(config_dir_name: str) -> None:
    """Import Model configurations as a python module.

    Args:
        config_dir_name (str): Path to model root folder.
    """

    if "nnet_project" in sys.modules.keys():
        submodules = [
            mod for mod in sys.modules.keys() if mod.startswith("nnet_project.")
        ]
        del sys.modules["nnet_project"]
        for submodule in submodules:
            del sys.modules[submodule]

    module_path = os.path.join(f"./models/{config_dir_name}", "__init__.py")
    module_path = os.path.abspath(module_path)

    spec = importlib.util.spec_from_file_location("nnet_project", module_path)

    module = importlib.util.module_from_spec(spec)
    sys.modules["nnet_project"] = module
    spec.loader.exec_module(module)
