import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import ndimage

def load_image(filename):
    image = plt.imread(filename)
    # input image is color png depicting grayscale, just use first plane from here on
    image = image[:, :, 1].astype(np.float64)
    # after reading and converting to float, image is between 0 and 1, we want to compute within the range 0 and 255
    image *= 255.0
    #print(image.shape)
    #print('image: min = ', np.min(image), ' max = ', np.max(image))
    return image

def pad_image(image):
    # compute required padding for later rotation: we assume image has same width and height
    padded_image_size = int(np.floor(image.shape[0] * np.sqrt(2)))
    # make sure padded image has odd dimensions
    if np.remainder(padded_image_size, 2) == 0:
        padded_image_size += 1

    # padding width is an integer since input size and padded size are both odd
    padding_width = int((padded_image_size - image.shape[0]) / 2)

    # place the input image into a padded version, extending with zeros
    padded_image = np.zeros((padded_image_size, padded_image_size), dtype=np.float64)
    padded_image[padding_width:image.shape[0] + padding_width,
                 padding_width:image.shape[1] + padding_width] = image

    return padded_image, padding_width

def calculate_sinogram(image, num_angles):
    image_size = image.shape[0]
    print(image_size)
    thetas = np.linspace(0., 180., num_angles, endpoint=False)
    # print(thetas.shape)
    # print(thetas)

    sinogram = np.zeros([num_angles, image_size], dtype=np.float64)

    for i, theta in enumerate(thetas):
        # print(theta)
        # rotate image by angle, counterclockwise
        imgR = ndimage.rotate(image, theta, order=1, reshape=False)

        # compute projection along columns (axis == 0 means sum is along columns!)
        projection_line = np.sum(imgR, axis=0)
        # print(projection_line.shape)

        # put projection lines into rows of sinogram (one row per discretized angle)
        sinogram[i, :] = projection_line

    return sinogram


def main():

    filename = 'CTThoraxSlice257.png'

    # note: angles are always between 0 and 180 degree, more angles are not necessary since projections would be redundant
    # if num_of_angle_steps is 360, we will therefore have a discretization into 360 angle steps between 0 and 180 degree (every half angle is used)
    # you are supposed to experiment with this number to simulate lower numbers of projection lines!
    num_of_angle_steps = 360

    # load and pad image
    image = load_image(filename)
    padded_image, padding_width = pad_image(image)
    print('padding width: ', padding_width)

    # calculate and save sinogram
    sinogram = calculate_sinogram(padded_image, num_of_angle_steps)
    sio.savemat('sinogram.mat', { 'sinogram': sinogram })
    
    print('sinogram: min = ', np.min(sinogram), ' max = ', np.max(sinogram))
    
    fig, axarr = plt.subplots(2, 2)

    axarr[0, 0].imshow(image, cmap=plt.cm.gray)
    axarr[0, 1].imshow(padded_image, cmap=plt.cm.gray)
    axarr[1, 0].imshow(sinogram, cmap=plt.cm.gray)
    axarr[1, 0].set_ylabel('Angles')
    axarr[1, 0].set_xlabel('Projected Values')

    plt.show()

if __name__ == "__main__":
    main()