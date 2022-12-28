"""Script to load and display a model summary"""
import argparse

from project.utils.set_config import set_import


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--model_name",
        metavar="model_name",
        type=str,
        help="Folder directory name that will be used to define the experiment configurations.",
    )

    parser.add_argument(
        "-v",
        "--view_summary",
        action="store_true",
        help="Print Keras model summary.",
    )

    args = parser.parse_args()

    set_import(config_dir_name=args.model_name)

    from project.utils.load_model import get_nnet_model

    nnet = get_nnet_model(config_dir_name=args.model_name, summary=args.view_summary)
