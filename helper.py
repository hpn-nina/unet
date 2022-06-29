# This helper will contain function for helping with the data augmentation problem
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy
import cv2
from configs import SEED, RESIZED_SIZE_HEIGHT, RESIZED_SIZE_WIDTH, PROB_AUGMENT, NUM_EPOCHS
from datetime import datetime
tf.random.set_seed(SEED)


def resize(input_img, input_mask):
    """
    Resize img to shape (RESIZED_SIZE_WIDTH, RESIZED_SIZE_HEIGHT)
    :param input_img: An image
    :param input_mask: An image mask
    :return: images with the same shape
    """
    input_img = tf.image.resize(input_img, (RESIZED_SIZE_WIDTH, RESIZED_SIZE_HEIGHT), method="nearest")
    input_mask = tf.image.resize(input_mask, (RESIZED_SIZE_WIDTH, RESIZED_SIZE_HEIGHT), method="nearest")
    return input_img, input_mask


def augment(input_img, input_mask):
    """
    With a probability random flip the image in horizontal vertices
    :param input_img: An image
    :param input_mask: An image mask
    :return: images with the same shape
    """
    if tf.random.uniform(()) > PROB_AUGMENT:
        return tf.image.flip_left_right(input_img), tf.image.flip_left_right(input_mask)
    return input_img, input_mask

def normalize(input_img, input_mask):
    """
    Convert all px of image into range (-1, 1) while input_mask decrease by 1
    :param input_img:
    :param input_mask:
    :return:
    """
    input_mask -= 1
    return tf.cast(input_img, tf.float32) / 255.0, input_mask


def apply_training(input_image, input_mask):
    input_image, input_mask = resize(input_image, input_mask)
    input_image, input_mask = augment(input_image, input_mask)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask


def apply_testing(input_image, input_mask):
    input_image, input_mask = resize(input_image, input_mask)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask


def load_image(datapoint, load_type="train"):
    input_image = datapoint["image"]
    input_mask = datapoint["segmentation_mask"]
    if load_type == "train":
        return apply_training(input_image, input_mask)
    else:
        return apply_testing(input_image, input_mask)

def load_image_train(datapoint):
    input_image = datapoint["image"]
    input_mask = datapoint["segmentation_mask"]
    input_image, input_mask = resize(input_image, input_mask)
    input_image, input_mask = augment(input_image, input_mask)
    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask 

def load_image_test(datapoint):
    input_image = datapoint["image"]
    input_mask = datapoint["segmentation_mask"]
    input_image, input_mask = resize(input_image, input_mask)
    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    plt.show()

def display_learning_curves(history, fig_name):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(NUM_EPOCHS)

    fig = plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    plt.plot(epochs_range, acc, label="train accuracy")
    plt.plot(epochs_range, val_acc, label="validataion accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")

    plt.subplot(1,2,2)
    plt.plot(epochs_range, loss, label="train loss")
    plt.plot(epochs_range, val_loss, label="validataion loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    fig.tight_layout()
    fig_name ='figure/history/' + fig_name + '.png'
    plt.savefig(fig_name)
    plt.show()

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(unet_model, dataset=None, num=1):
    fig, axs = plt.subplots(3,3)
    
    title = ["Input Image", "True Mask", "Predicted Mask"]
    x, y = -1, -1
    for image, mask in dataset.take(num):
        x += 1
        pred_mask = unet_model.predict(image)
        display_list = [image[0], mask[0], create_mask(pred_mask)]

        for i in range(len(display_list)):
            y += 1
            if x == 0:
                axs[x,y].set_title(title[i])
            axs[x,y].imshow(tf.keras.utils.array_to_img(display_list[i]))
            
        y = -1

    name = 'test_figure_' + datetime.now().strftime("%H_%M_%S_%d_%m_%Y") + '.png'
    plt.axis("off")
    plt.savefig(name)
    plt.show()
