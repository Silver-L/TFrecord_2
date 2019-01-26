# coding:utf-8
import os
import numpy as np
import SimpleITK as sitk

def read_mhd_and_raw(path, numpyFlag=True):
    """
    This function use sitk
    path : Meta data path
    ex. /hogehoge.mhd
    numpyFlag : Return numpyArray or sitkArray
    return : numpyArray(numpyFlag=True)
    Note ex.3D :numpyArray axis=[z,y,x], sitkArray axis=(z,y,x)
    """
    img = sitk.ReadImage(path)
    if not numpyFlag:
        return img

    nda = sitk.GetArrayFromImage(img)  # (img(x,y,z)->numpyArray(z,y,x))
    return nda


def write_mhd_and_raw(Data, path):
    """
    This function use sitk
    Data : sitkArray
    path : Meta data path
    ex. /hogehoge.mhd
    """
    if not isinstance(Data, sitk.SimpleITK.Image):
        print('Please check your ''Data'' class')
        return False

    data_dir, file_name = os.path.split(path)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    sitk.WriteImage(Data, path, True)

    return True

def load_matrix_data(path, dtype):
    # load data_list
    data_list = []

    with open(path) as paths_file:
        for line in paths_file:
            line = line.split()
            if not line: continue
            data_list.append(line[:])

    data = []
    for i in data_list:
        print('image from: {}'.format(i[0]))

        image = read_mhd_and_raw(i[0]).astype(dtype)
        data.append(image)

    data = np.asarray(data)
    return data

# load list
def load_list(path):
    data_list = []
    with open(path) as paths_file:
        for line in paths_file:
            if not line: continue
            line = line.replace('\n','')
            data_list.append(line[:])
    return data_list


def read_raw(path, dtype):
    """
    path : input image name
    ex. /hoge.raw
    dtype : type of data
    ex. 'float' or 'np.float32' or 'MET_FLOAT'
    return : numpy array
    ----------------------
    np.int8      or char
    np.uint8     or uchar
    np.int16     or short
    np.int32     or int
    np.uint32    or long
    np.float32   or float
    np.float64   or double
    ----------------------
    """
    type = __change2NP(dtype)
    data = np.fromfile(path, type)

    return data

def __change2NP(type):
    """
    return : numpy data type
    type : type of data
    ----------------------
    np.int8      or char   or MET_CHAR
    np.int16     or short  or MET_SHORT
    np.int32     or int    or MET_INT
    np.float32   or float  or MET_FLOAT
    np.float64   or double or MET_DOUBLE
    np.uint8     or uchar  or MET_UCHAR
    np.uint16    or ushort or MET_USHORT
    np.uint32    or uint   or MET_UINT
    ----------------------
    """
    if isinstance(type, str):
        if (type == "char") or (type == "MET_CHAR"):
            return np.int8
        elif (type == "short") or (type == "MET_SHORT"):
            return np.int16
        elif (type == "int") or (type == "MET_INT"):
            return np.int32
        elif (type == "float") or (type == "MET_FLOAT"):
            return np.float32
        elif (type == "double") or (type == "MET_DOUBLE"):
            return np.float64
        elif (type == "uchar") or (type == "MET_UCHAR"):
            return np.uint8
        elif (type == "ushort") or (type == "MET_USHORT"):
            return np.uint16
        elif (type == "uint") or (type == "MET_UINT"):
            return np.uint32
        else:
            print("korakora!")
            quit()
    else:
        if (type == np.int8):
            return np.int8
        elif (type == np.int16):
            return np.int16
        elif (type == np.int32):
            return np.int32
        elif (type == np.float32):
            return np.float32
        elif (type == np.float64):
            return np.float64
        elif (type == np.uint8):
            return np.uint8
        elif (type == np.uint16):
            return np.uint16
        elif (type == np.uint32):
            return np.uint32
        else:
            print("korakora!")
            quit()


def load_matrix_raw_data(path, dtype):
    # load data_list
    data_list = []

    with open(path) as paths_file:
        for line in paths_file:
            line = line.split()
            if not line: continue
            data_list.append(line[:])

    data = []
    for i in data_list:
        print('image from: {}'.format(i[0]))

        image = read_raw(i[0], dtype=dtype)
        data.append(image)

    data = np.asarray(data)
    return data