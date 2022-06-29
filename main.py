import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from helper import load_image, load_image_test, load_image_train, display, display_learning_curves, show_predictions
from model import build_unet_model
from configs import BUFFER_SIZE, BATCH_SIZE, NUM_EPOCHS, VAL_SUBSPLITS, loss, metrics, optimizer, PREDICTION_NUM, SHOW_MASK_PREDICTION, SHOW_MODEL_HISTORY


# load dataset
print('loading dataset...')
dataset, info = tqdm(tfds.load('oxford_iiit_pet:3.*.*', with_info=True))

#print(info)
#print(dataset)
#print(dataset["train"])

# preprocess data
print('preprocessing data...')
train_dataset = dataset["train"].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = dataset["test"].map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)

# create train, val, test batches
print('creating train, val, test batches...')
train_batches = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_batches = train_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
validation_batches = test_dataset.take(3000).batch(BATCH_SIZE)
test_batches = test_dataset.skip(3000).take(669).batch(BATCH_SIZE)

# visualize data
print('visualizing data...')
sample_batch = next(iter(test_batches))
random_index = np.random.choice(sample_batch[0].shape[0])
sample_image, sample_mask = sample_batch[0][random_index], sample_batch[1][random_index]
display([sample_image, sample_mask])

# build model
print('building model...')
unet_model = build_unet_model()
unet_model.summary()
tf.keras.utils.plot_model(unet_model, show_shapes=True)

# compile and train u-net
print('compiling model...')
unet_model.compile(optimizer=optimizer,
                   loss=loss,
                   metrics=metrics)

TRAIN_LENGTH = info.splits["train"].num_examples
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

TEST_LENGTH = info.splits["test"].num_examples
VALIDATION_STEPS = TEST_LENGTH // BATCH_SIZE // VAL_SUBSPLITS

print('training model...')
model_history = unet_model.fit(train_batches,
                                epochs=NUM_EPOCHS,
                                steps_per_epoch=STEPS_PER_EPOCH,
                                validation_steps=VALIDATION_STEPS,
                                validation_data=validation_batches)

if SHOW_MODEL_HISTORY:
    print('done training, saving figure...')
    fig_name = 'model_history_' + datetime.now().strftime("%H_%M_%S__%d_%m_%Y")
    display_learning_curves(unet_model.history, fig_name)      

if SHOW_MASK_PREDICTION:
    count = 0
    for i in test_batches:
        count +=1
    print("number of batches:", count)

    show_predictions(test_batches.skip(5), PREDICTION_NUM)