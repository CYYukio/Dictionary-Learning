import matplotlib.pyplot as plt
import numpy as np
import pydicom
from sklearn import linear_model



def dict_update(y, d, x, n_components):
    for i in range(n_components):
        index = np.nonzero(x[i, :])[0]
        if len(index) == 0:
            continue
        # 更新第i列
        d[:, i] = 0
        # 计算误差矩阵
        r = (y - np.dot(d, x))[:, index]
        # 利用svd的方法，来求解更新字典和稀疏系数矩阵
        u, s, v = np.linalg.svd(r, full_matrices=False)
        # 使用左奇异矩阵的第0列更新字典
        d[:, i] = u[:, 0]
        # 使用第0个奇异值和右奇异矩阵的第0行的乘积更新稀疏系数矩阵
        for j,k in enumerate(index):
            x[i, k] = s[0] * v[0, j]
    return d, x

if __name__ == '__main__':
    dicom1 = pydicom.read_file('L067_FD_1_1.CT.0001.0001.2015.12.22.18.09.40.840353.358074219.IMA')
    image = dicom1.pixel_array
    print('read: ', image.shape)

    u, s, v = np.linalg.svd(image)
    n_comp = 1024
    dict_data = np.append(u[:, :512], u[:, :512], axis=1)
    print('dict_dim: ',dict_data.shape)
    max_iter = 5
    dictionary = dict_data

    y = image
    tolerance = 1e-6

    for i in range(max_iter):
        # 稀疏编码
        x = linear_model.orthogonal_mp(dictionary, y)
        e = np.linalg.norm(y - np.dot(dictionary, x))
        if e < tolerance:
            break
        dict_update(y, dictionary, x, n_comp)
        print('round: ',i)

    sparsecode = linear_model.orthogonal_mp(dictionary, y)

    train_restruct = dictionary.dot(sparsecode)

    print('dictionary: ',dictionary.shape)
    print('sparsecode: ',sparsecode.shape)

    plt.imshow(train_restruct, cmap='gray')
    plt.show()

