from pickletools import optimize
import tensorflow as tf

# Define constant variables
SEED = 19521020
PROB_AUGMENT = 0.5
RESIZED_SIZE_WIDTH = 128
RESIZED_SIZE_HEIGHT = 128

BATCH_SIZE = 16
BUFFER_SIZE = 1000
NUM_EPOCHS = 20
VAL_SUBSPLITS = 5

loss = "sparse_categorical_crossentropy"
metrics = "accuracy"
optimizer = tf.keras.optimizers.Adam()

SHOW_MASK_PREDICTION = True
PREDICTION_NUM = 3
SHOW_MODEL_HISTORY = True