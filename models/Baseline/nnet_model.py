"""Model configurations"""


# Custom operations
from .keras_custom_layers import Partition2D, Seq_Slicer


from .nnet_metrics import (
    soft_dice_loss,
    hard_dice,
    precision,
    recall,
)


def custom_obj():
    """Import custom objects used during the model experiment."""
    from keras.utils.generic_utils import get_custom_objects

    get_custom_objects()["Seq_Slicer"] = Seq_Slicer
    get_custom_objects()["dsc_2D"] = hard_dice
    get_custom_objects()["soft_dice_loss_2D"] = soft_dice_loss
    get_custom_objects()["prec_2D"] = precision
    get_custom_objects()["rec_2D"] = recall
    get_custom_objects()["Partition2D"] = Partition2D
