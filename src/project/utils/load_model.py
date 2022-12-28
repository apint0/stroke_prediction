"""Tools to load a Keras-based model."""
import os
from keras.models import load_model
from project.utils.set_config import set_import

from nnet_project import nnet_model


def get_nnet_model(
    config_dir_name: str, summary: bool = False, compile: bool = False
) -> tuple:
    """Load a saved model along with the respective callbacks.

    Args:
        config_dir_name (str): Path to the model's root dir.
        summary (bool, optional): Whether to print a model summary or not. Defaults to False.
        compile (bool, optional): Keras flag to compile the model. Defaults to False.

    Returns:
        tuple: Loaded network model and the respective callbacks.
    """

    if hasattr(nnet_model, "custom_obj"):
        nnet_model.custom_obj()

    nnet = load_model(
        os.path.abspath(f"models/{config_dir_name}/states/nnet-state.h5"),
        compile=compile,
    )

    if summary:
        print(nnet.summary())

    return nnet
