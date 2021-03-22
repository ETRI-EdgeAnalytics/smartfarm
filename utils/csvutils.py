import numpy as np
import tensorflow as tf

__all__ = ['tensor2csv']

def tensor2csv(filename, tensor):
    np_tensor = tensor.numpy()
    np_shape = np_tensor.shape

    if np_shape[-1] == 1:
        np_array = np_tensor.reshape((np_shape[1], np_shape[2]))
        np.savetxt(filename, np_array, fmt='%f', delimiter=' ')
    else:
        np_tensor_num = np_tensor[:, :, :, 0]
        np_tensor_kg = np_tensor[:, :, :, 1]
        fn = filename.split('.')

        np_array1 = np_tensor_num.reshape((np_shape[1], np_shape[2]))
        np.savetxt(fn[0]+'_num.csv', np_array1, fmt='%f', delimiter=' ')

        np_array2 = np_tensor_kg.reshape((np_shape[1], np_shape[2]))
        np.savetxt(fn[0]+'_kg.csv', np_array2, fmt='%f', delimiter=' ')
