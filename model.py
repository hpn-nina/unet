

def double_conv_block(x, n_filters):
    """
    This block include COnv2D-RELU-Conv2D-RELU
    :param x:
    :param n_filters:
    :return:
    """
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
