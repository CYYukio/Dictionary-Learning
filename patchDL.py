from keras.preprocessing.image import load_img,img_to_array
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d,reconstruct_from_patches_2d
from sklearn.decomposition import MiniBatchDictionaryLearning
import pydicom
import os
def lena256():
    image = load_img('./lena256.tif')
    img = img_to_array(image)

    img = img[:, :, 0]
    print('origin shape: ',img.shape)

    # image = image.astype('float32')
    img /= 255.0
    # print(img.shape, type(img[0][0]))
    # plt.imshow(image, cmap='gray')
    # plt.show()

    noise = np.random.normal(loc=0, scale=0.05, size=img.shape)
    x_test_noisy1 = img + noise
    x_test_noisy1 = np.clip(x_test_noisy1, 0., 1.)

    # plt.imshow(x_test_noisy1, cmap='gray')
    # plt.show()

    print('Extracting reference patches...')
    patch_size = (5, 5)
    data = extract_patches_2d(img, patch_size)
    print(data.shape)

    data = data.reshape(data.shape[0], -1)  # patch变成一行
    print(data.shape)

    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0)

    # #############################################################################
    # Learn the dictionary from reference patches
    print('Learning the dictionary...')
    dico = MiniBatchDictionaryLearning(n_components=144, alpha=1, n_iter=500)
    V = dico.fit(data).components_

    print(V.shape)

    '''
    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(V[:144]):
        plt.subplot(12, 12, i + 1)
        plt.imshow(comp.reshape(patch_size), cmap='gray')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('Dictionary learned from patches\n', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.show()
    '''
    # #############################################################################
    # Extract noisy patches and reconstruct them using the dictionary
    print('Extracting noisy patches... ')
    noisedata = extract_patches_2d(x_test_noisy1, patch_size)
    noisedata = noisedata.reshape(data.shape[0], -1)
    intercept = np.mean(noisedata, axis=0)
    noisedata -= intercept


    print('Orthogonal Matching Pursuit\n2 atoms' + '...')
    reconstructions = x_test_noisy1.copy()
    dico.set_params(transform_algorithm='omp', **{'transform_n_nonzero_coefs': 2})
    code = dico.transform(noisedata)
    patches = np.dot(code, V)
    print('code:', code.shape, 'dic', V.shape)

    patches += intercept
    patches = patches.reshape(len(noisedata), *patch_size)

    reconstructions = reconstruct_from_patches_2d(patches, (256, 256))


    plt.subplot(3,1,1)
    plt.imshow(img,cmap='gray')
    plt.subplot(3, 1, 2)
    plt.imshow(x_test_noisy1, cmap='gray')
    plt.subplot(3,1,3)
    plt.imshow(reconstructions, cmap='gray')
    plt.show()


if __name__ == '__main__':
    lena256()