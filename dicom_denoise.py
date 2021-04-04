from keras.preprocessing.image import load_img,img_to_array
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d,reconstruct_from_patches_2d
from sklearn.decomposition import MiniBatchDictionaryLearning
import pydicom
import os
from skimage.transform.radon_transform import radon, iradon
def preprocess():
    if not os.path.exists('cz_sino.npz'):
        czsino = np.load('./interpolation_sinoX8.npz')['arr_0']
        image = czsino[0].reshape((512, 512))
        np.savez_compressed('cz_sino.npz', image)
        print('interpolation sino extracted')

    if not os.path.exists('full_sino.npz'):
        fsino = np.load('./full_sinoX8.npz')['arr_0']
        image = fsino[0].reshape((512, 512))
        np.savez_compressed('full_sino.npz', image)
        print('full sino extracted')

def DL(img1 ,img2):
    img1 = img1 /1.0
    img2 = img2 / 1.0
    print(img1.shape, img2.shape)

    print('Extracting reference patches...')
    patch_size = (8, 8)
    data = extract_patches_2d(img2, patch_size)
    print(data.shape)
    data = data.reshape(data.shape[0], -1)  # patch变成一行
    print(data.shape)
    data -= np.mean(data, axis=0)

    # #############################################################################
    # Learn the dictionary from reference patches
    print('Learning the dictionary...')
    dico = MiniBatchDictionaryLearning(n_components=256, alpha=1, n_iter=1000)
    V = dico.fit(data).components_
    print(V.shape)

    # #############################################################################
    # Extract noisy patches and reconstruct them using the dictionary
    print('Extracting noisy patches... ')
    noisedata = extract_patches_2d(img1, patch_size)
    noisedata = noisedata.reshape(data.shape[0], -1)
    intercept = np.mean(noisedata, axis=0)
    noisedata -= intercept

    print('Orthogonal Matching Pursuit\n2 atoms' + '...')
    reconstructions = img1.copy()
    dico.set_params(transform_algorithm='omp', **{'transform_n_nonzero_coefs': 2})
    code = dico.transform(noisedata)
    patches = np.dot(code, V)
    print('code:', code.shape, 'dic', V.shape)

    patches += intercept
    patches = patches.reshape(len(noisedata), *patch_size)

    reconstructions = reconstruct_from_patches_2d(patches, (512, 512))

    sino_list = list()
    sino_list.append(img1)
    sino_list.append(img2)
    sino_list.append(reconstructions)
    np.savez_compressed('result2.npz', sino_list)

    plt.subplot(3, 1, 1)
    plt.imshow(img2, cmap='gray')
    plt.subplot(3, 1, 2)
    plt.imshow(img1, cmap='gray')
    plt.subplot(3, 1, 3)
    plt.imshow(reconstructions, cmap='gray')
    plt.show()

def test():
    sino_list = np.load('./result.npz')['arr_0']

    czsino = sino_list[0]
    fsino = sino_list[1]
    rsino = sino_list[2]
    theta = np.linspace(0, 180, 512, endpoint=False)
    czimage = iradon(czsino,theta,circle=True)
    fimage = iradon(fsino, theta, circle=True)
    rimage = iradon(rsino, theta, circle=True)


    plt.subplot(3, 2, 1)
    plt.title('cz')
    plt.imshow(czsino, cmap='gray')
    plt.subplot(3, 2, 2)
    plt.title('cz')
    plt.imshow(czimage, cmap='gray')

    plt.subplot(3, 2, 3)
    plt.title('full')
    plt.imshow(fsino, cmap='gray')
    plt.subplot(3, 2, 4)
    plt.title('full')
    plt.imshow(fimage, cmap='gray')

    plt.subplot(3, 2, 5)
    plt.title('inpainting')
    plt.imshow(rsino, cmap='gray')
    plt.subplot(3, 2, 6)
    plt.title('inpainting')
    plt.imshow(rimage, cmap='gray')
    plt.show()


def dicom_denoise():
    fullimg = pydicom.read_file('L067_FD_1_1.CT.0001.0001.2015.12.22.18.09.40.840353.358074219.IMA').pixel_array

    sparseimg = pydicom.read_file('L067_QD_1_1.CT.0003.0001.2015.12.22.18.10.55.420810.358276339.IMA').pixel_array

    DL(sparseimg, fullimg)


if __name__ == '__main__':
    preprocess()
    # DL()
    # test()
    dicom_denoise()
