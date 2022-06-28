# This helper will contain function for helping with the data augmentation problem
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy
import cv2

tf.random.set_seed(19520208)

# Define constant variables
PROB_AUGMENT = 0.5
RESIZED_SIZE_WIDTH = 128
RESIZED_SIZE_HEIGHT = 128


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

def test_function(_):
    print('import ok')



def display(list_display):
    plt.figure(figsize=(15, 15))
    title = ["Input Image", "True Mask", "Predicted Mask"]

    for (i, image) in zip(enumerate(len(list_display)), list_display):
        plt.subplot(1, len(list_display), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(image))
        plt.axis("off")

    plt.show()
