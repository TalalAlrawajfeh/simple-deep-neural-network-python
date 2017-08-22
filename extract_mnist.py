#!/usr/bin/python3

"""
    Created by Talal Alrawajfeh on 09/22/2017
"""

import struct

import cv2
import numpy as np

import mnist_data_reader


def extract_labels(mnist_label_file_path, label_file_path):
    with open(mnist_label_file_path, "rb") as mnist_label_file:
        mnist_label_file.read(mnist_data_reader.LABELS_OFFSET)
        label_file = open(label_file_path, "w")
        label = mnist_label_file.read(1)
        while label:
            label_file.writelines(str(label[0]) + "\n")
            label = mnist_label_file.read(1)
        label_file.close()


def extract_images(images_file_path, images_save_folder):
    with open(images_file_path, "rb") as images_file:
        images_file.read(mnist_data_reader.IMAGES_OFFSET)
        count = 1
        image = np.zeros((mnist_data_reader.IMAGE_PIXEL_ROWS, mnist_data_reader.IMAGE_PIXEL_COLUMNS, 1), np.uint8)
        image_bytes = images_file.read(mnist_data_reader.IMAGE_SIZE)
        while image_bytes:
            image_unsigned_char = struct.unpack("=784B", image_bytes)
            for i in range(mnist_data_reader.IMAGE_SIZE):
                image.itemset(i, image_unsigned_char[i])
            image_save_path = "./%s/%d.png" % (images_save_folder, count)
            cv2.imwrite(image_save_path, image)
            print(count)
            image_bytes = images_file.read(mnist_data_reader.IMAGE_SIZE)
            count += 1

if __name__ == '__main__':
    extract_images('./resources/train-images.idx3-ubyte', './out/images')