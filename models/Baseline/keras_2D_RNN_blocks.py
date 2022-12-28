""" Recurrent NN layers to be used in an image-based input"""

from __future__ import print_function

from keras import backend as K

__COMPILER__ = K.image_dim_ordering()


from keras.layers.core import Permute
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed, Bidirectional

from .keras_custom_layers import Partition2D


def rnn_layer_init(
    units: int,
    dropout: float,
    return_sequences: bool = True,
    implementation: int = 2,
    initializer: any = None,
    regularizer: any = None,
) -> callable:
    """Configure recurrent unit block, as a TimeDistributed bidirectional layer to be applicable to batches of images.

    Args:
        units (int): Positive integer, dimensionality of the output space.
        dropout (float): Float between 0 and 1.
            Fraction of the units to drop for the linear transformation of the inputs.
        return_sequences (bool, optional): Whether to return the last output
            in the output sequence, or the full sequence. Defaults to True.
        implementation (int, optional): Mode 1 will structure its operations as a larger number of smaller dot products
            and additions, whereas mode 2 will batch them into fewer, larger operations. These modes will have different
            performance profiles on different. Defaults to 2.
        initializer (any, optional): Initializer for the non-recurrent and recurrent kernels weights matrix.
            Defaults to None.
        regularizer (any, optional): Regularizer for the non-recurrent and recurrent kernels weights matrix.
            Defaults to None.

    Returns:
        callable: TimeDistributed object
    """

    # Initing/Regularizing the linear an recurrent part equally
    rnn_layer = LSTM(
        units=units,
        dropout=dropout,
        return_sequences=return_sequences,
        implementation=implementation,
        kernel_initializer=initializer,
        recurrent_initializer=initializer,
        kernel_regularizer=regularizer,
        recurrent_regularizer=regularizer,
    )

    rnn_layer = TimeDistributed(Bidirectional(rnn_layer))
    return rnn_layer


def renet(
    input_data: callable,
    dropout: float,
    hidden_layers: int,
    patch_size: int,
    initializer=None,
    regularizer=None,
) -> tuple:
    """Bidimensional recurrent layer block.

    Args:
        input_data (callable): Tensor object.
        dropout (float): Float between 0 and 1.
            Fraction of the units to drop for the linear transformation of the inputs.
        hidden_layers (int): Positive integer, dimensionality of the output space.
        patch_size (int): W and H dimensions of the patch.
        initializer (any, optional): Initializer for the non-recurrent and recurrent kernels weights matrix.
            Defaults to None.
        regularizer (any, optional): Regularizer for the non-recurrent and recurrent kernels weights matrix.
            Defaults to None.

    Returns:
        out_shape (tuple): Tensor output shape.
        out (callable): Tensor object.
    """

    out = Partition2D(pool_size=(patch_size, patch_size), border_mode="valid")(input_data)

    h_rnn_block = rnn_layer_init(
        units=hidden_layers,
        dropout=dropout,
        initializer=initializer,
        regularizer=regularizer,
    )

    out = h_rnn_block(out)
    out = Permute((2, 1, 3))(out)

    v_rnn_block = rnn_layer_init(
        units=hidden_layers,
        initializer=initializer,
        dropout=dropout,
        regularizer=regularizer,
    )
    out = v_rnn_block(out)

    if __COMPILER__ == "th":
        permutation_layer = Permute((3, 2, 1))
    elif __COMPILER__ == "tf":
        permutation_layer = Permute((2, 1, 3))

    out = permutation_layer(out)
    out_shape = permutation_layer.output_shape[1:]

    return out_shape, out
