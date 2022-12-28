"""Custom Layers for Keras"""

from keras.engine import Layer, InputSpec
from keras.utils.conv_utils import conv_output_length
from keras import backend as K

__COMPILER__ = K.image_dim_ordering()

if __COMPILER__ == "tf":
    from tensorflow import extract_image_patches
else:
    from theano.tensor.nnet.neighbours import images2neibs
    import theano.tensor as T


class Seq_Slicer(Layer):
    """Class of type Keras Layer that splits the input tensor by discarding the last dimension. Usefull when splitting
    the input data into different streams of processing blocks."""

    def __init__(self, dim_ordering: str = "default", **kwargs):
        """Class constructor

        Args:
            dim_ordering (str, optional): Whether to use the dim ordering from keras configuration file or force a
                specific ordering. Defaults to "default".
        """
        super(Seq_Slicer, self).__init__(**kwargs)

        if dim_ordering == "default":
            self.dim_ordering = K.image_dim_ordering()
        else:
            self.dim_ordering = dim_ordering

    def build(self, input_shape: list):
        """Layer build useful for model compilation.

        Args:
            input_shape (list): List specifying the input dimensions.
        """

        super(Seq_Slicer, self).build(input_shape)

    def call(self, x):
        """Function call

        Args:
            x (tensor): 4+D tensor with shape: `batch_shape + (channels, rows, cols)` if
        `data_format='channels_first'`

        Returns:
            tensor: 4+D tensor with the last channel removed from the respective dimension.
        """
        if self.dim_ordering == "th":
            return x[:, :-1, :, :]
        elif self.dim_ordering == "tf":
            return x[:, :, :, :-1]

    def compute_output_shape(self, input_shape: list):
        """Keras base function to compute output shape at model compilation.

        Args:
            input_shape (list): List of input dimensions

        Returns:
            list: Output dimensions.
        """
        if self.dim_ordering == "th":
            return (input_shape[0], input_shape[1] - 1, input_shape[2], input_shape[3])
        elif self.dim_ordering == "tf":
            return (input_shape[0], input_shape[1], input_shape[2], input_shape[3] - 1)


class Partition2D(Layer):
    """Class that allows the partition of a 4+D tensor into several 'temporal' blocks ought to be processed by the
    recurrent units.

    Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.
    Output shape
        5D tensor with shape:
        `(nb_samples, channels, nb_small_patches, new_row_dim, new_col_dim)`
        if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, nb_small_patches, new_row_dim, new_col_dim, channels)`
        if dim_ordering='tf'.
    """

    def __init__(
        self,
        pool_size: tuple = (2, 2),
        strides: int = None,
        border_mode: str = "valid",
        dim_ordering: str = "default",
        **kwargs
    ):
        """Class constructor.

        Args:
            pool_size (tuple, optional): 2D dimension of each 'temporal' block. Defaults to (2, 2).
            strides (int, optional): Striding configuration. Defaults to None.
            border_mode (str, optional): _description_. Defaults to "valid".
            dim_ordering (str, optional): _description_. Defaults to "default".
        """

        super(Partition2D, self).__init__(**kwargs)

        if dim_ordering == "default":
            dim_ordering = K.image_dim_ordering()
        self.pool_size = tuple(pool_size)

        if strides is None:
            strides = self.pool_size
        self.strides = tuple(strides)

        assert border_mode in {
            "valid",
            "ignore_borders",
        }, "border_mode must be in {valid, same}"
        self.border_mode = border_mode

        assert dim_ordering in {"tf", "th"}, "dim_ordering must be in {tf, th}"
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=4)]

    def build(self, input_shape: list):
        """Layer build useful for model compilation.

        Args:
            input_shape (list): List specifying the input dimensions.
        """

        super(Partition2D, self).build(input_shape)

    def compute_output_shape(self, input_shape: list):
        """Keras base function to compute output shape at model compilation.

        Args:
            input_shape (list): List of input dimensions

        Returns:
            list: Output dimensions.
        """

        if self.dim_ordering == "th":
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == "tf":
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            raise Exception("Invalid dim_ordering: " + self.dim_ordering)

        rows = conv_output_length(rows, self.pool_size[0], self.border_mode, self.strides[0])
        cols = conv_output_length(cols, self.pool_size[1], self.border_mode, self.strides[1])

        if self.dim_ordering == "th":
            self.output_dims = (
                input_shape[0],
                rows,
                cols,
                self.strides[0] * self.strides[1] * input_shape[1],
            )

            return self.output_dims

        elif self.dim_ordering == "tf":
            self.output_dims = (
                input_shape[0],
                rows,
                cols,
                self.strides[0] * self.strides[1] * input_shape[3],
            )
            return self.output_dims
        else:
            raise Exception("Invalid dim_ordering: " + self.dim_ordering)

    def call(self, x):
        """Function call

        Args:
            x (tensor): 4+D tensor with shape: `batch_shape + (channels, rows, cols)` if
        `data_format='channels_first'`

        Returns:
            tensor: Partioned temporal data ready to be processed temporally.
        """

        if self.dim_ordering == "tf":

            output = extract_image_patches(
                x,
                ksizes=[1, self.pool_size[0], self.pool_size[1], 1],
                strides=[1, self.strides[0], self.strides[1], 1],
                rates=[1, 1, 1, 1],
                padding=self.border_mode.upper(),
            )

        elif self.dim_ordering == "th":

            x_shape = K.shape(x)

            num_rows = 1 + ((x_shape[-1] - self.pool_size[0]) // self.pool_size[0])
            num_cols = 1 + ((x_shape[-2] - self.pool_size[1]) // self.pool_size[1])
            ch = x_shape[-3]
            p = self.pool_size[1]

            patches = images2neibs(
                x,
                neib_shape=self.pool_size,
                neib_step=self.strides,
                mode=self.border_mode,
            )
            # Reshaping to output (bs, nr, nc, ch*p*p)

            patches = K.reshape(patches, (-1, ch, num_rows, num_cols, p * p))
            patches = K.permute_dimensions(patches, (0, 2, 3, 1, 4))
            output = K.reshape(patches, (-1, num_rows, num_cols, ch * p * p))

        return output

    def get_config(self):
        """Base operation of Keras layer to define Partition2D configurations.

        Returns:
            dict: Configurations.
        """

        config = {
            "pool_size": self.pool_size,
            "border_mode": self.border_mode,
            "strides": self.strides,
            "dim_ordering": self.dim_ordering,
        }

        base_config = super(Partition2D, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
