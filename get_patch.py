import numpy as np
import pydicom
import matplotlib.pyplot as plt


def get_patch(img, index):
    patch = np.zeros((3, 3))
    row = img.shape[0]
    col = img.shape[1]

    x = index[0]
    y = index[1]

    for i in range(3):
        for j in range(3):
            patch[i][j] = img[x+i][y+j]

    print('get patch', patch)

    return patch


if __name__ == '__main__':
    dicom1 = pydicom.read_file('L067_FD_1_1.CT.0001.0001.2015.12.22.18.09.40.840353.358074219.IMA')
    image = dicom1.pixel_array

    img = np.zeros((3, 3))
    img[0][0] = image[0][0]
    img[0][1] = image[0][1]
    img[0][2] = image[0][2]
    img[1][0] = image[1][0]
    img[1][1] = image[1][1]
    img[1][2] = image[1][2]
    img[2][0] = image[2][0]
    img[2][1] = image[2][1]
    img[2][2] = image[2][2]
    print('real patch', img)

    patch = get_patch(img, (0, 0))

    plt.subplot(1, 2, 1)
    plt.imshow(patch, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.imshow(img, cmap='gray')

    plt.show()


