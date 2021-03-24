import matplotlib.pyplot as plt
import numpy as np
import pydicom

def SVD(image):
    image = image/1.0
    print('pixel len: ',type(image[0][0]))
    eval_sigma1, evec_u = np.linalg.eigh(image.dot(image.T))
    # 降序排列后，逆序输出
    eval1_sort_idx = np.argsort(eval_sigma1)[::-1]
    # 将特征值对应的特征向量也对应排好序
    eval_sigma1 = np.sort(eval_sigma1)[::-1]
    evec_u = evec_u[:, eval1_sort_idx]
    # 计算奇异值矩阵的逆
    eval_sigma1 = np.sqrt(eval_sigma1)
    eval_sigma1_inv = np.linalg.inv(np.diag(eval_sigma1))

    # 计算右奇异矩阵
    evec_part_v = eval_sigma1_inv.dot((evec_u.T).dot(image))
    return evec_u, eval_sigma1, evec_part_v

if __name__ == '__main__':
    dicom1 = pydicom.read_file('L067_FD_1_1.CT.0001.0001.2015.12.22.18.09.40.840353.358074219.IMA')
    image = dicom1.pixel_array
    print('read: ', image.shape)

    U, Sigma, VT=SVD(image)
    print('U: ', U.shape)
    print('Sigma: ', Sigma.shape)
    print('VT: ', VT.shape)

    sval_nums = 16
    img_restruct1 = (U[:, 0:sval_nums]).dot(np.diag(Sigma[0:sval_nums])).dot(VT[0:sval_nums, :])
    img_restruct1 = img_restruct1.reshape(512, 512)

    sval_nums = 64
    img_restruct2 = (U[:, 0:sval_nums]).dot(np.diag(Sigma[0:sval_nums])).dot(VT[0:sval_nums, :])
    img_restruct2 = img_restruct2.reshape(512, 512)

    plt.subplot(1, 3, 1)
    plt.title('groundtruth')
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title('16 SVD')
    plt.imshow(img_restruct1, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('64 SVD')
    plt.imshow(img_restruct2, cmap='gray')
    plt.show()



