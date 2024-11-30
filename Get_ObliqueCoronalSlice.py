import numpy as np
import scipy, shutil, os, nibabel
import sys, getopt
import imageio
import glob
import cv2
import pandas as pd
import math
import scipy.linalg as linalg
import nibabel as nib
import matplotlib.pyplot as plt
import torch


def loc_convert(loc, axis, radian):
    '''
	Realize a point rotated by a certain number of degrees on a certain axis and get the coordinates of the new point.
    :param loc: original coordinate
    :param axis: Axis of rotation (point)
    :param radian: Rotation angle
    :return: new coordinate
    '''
    radian = np.deg2rad(radian)
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
    new_loc = np.dot(rot_matrix, loc)
    return new_loc

def extract_slice(img, c, v, radius):
    '''
    :param img: Raw 3D data
    :param center-c: Get the center of the 2D slice you need extract（x,y,z）
    :param normal-v: Normal vector of the 2D slice（v1,v2,v3）
    :param radius: The radius of the 2D slice, which is half the length of the image's sides
    :return:
    slicer：Obtained 2D slices
    loc: Get the original 3d coordinates corresponding to the slice
    '''
    # Setting the Initial Plane
    epsilon = 1e-12
    x = np.arange(-radius, radius, 1)
    y = np.arange(-radius, radius, 1)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    loc = np.array([X.flatten(), Y.flatten(), Z.flatten()])

    # Set the initial plane, perpendicular to the YZ-plane, to change vectors to unit vectors
    hspInitialVector = np.array([1, 0, 0])
    h_norm = np.linalg.norm(hspInitialVector)
    h_v = hspInitialVector / h_norm
    h_v[h_v == 0] = epsilon
    v = v / np.linalg.norm(v)
    v[v == 0] = epsilon

    # Calculate the angle between the initial normal vector and the final normal vector
    hspVecXvec = np.cross(h_v, v) / np.linalg.norm(np.cross(h_v, v))
    acosineVal = np.arccos(np.dot(h_v, v))
    hspVecXvec[np.isnan(hspVecXvec)] = epsilon
    acosineVal = epsilon if np.isnan(acosineVal) else acosineVal

    # Get the coordinates after rotation
    loc = loc_convert(loc, hspVecXvec, 180 * acosineVal / math.pi)
    sub_loc = loc + np.reshape(c, (3, 1))
    loc = np.round(sub_loc)
    loc = np.reshape(loc, (3, X.shape[0], X.shape[1]))

    # Generate initial slices, and corresponding index values
    sliceInd = np.zeros_like(X, dtype=np.float)
    sliceInd[sliceInd == 0] = np.nan
    slicer = np.copy(sliceInd)

    # Assigns the corresponding pixel values and the corresponding coordinates of the 3D image to the corresponding slice
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if loc[0, i, j] >= 0 and loc[0, i, j] < img.shape[0] and loc[1, i, j] >= 0 and loc[1, i, j] < img.shape[1] and loc[2, i, j] >= 0 and loc[2, i, j] < img.shape[2]:
                slicer[i, j] = img[
                    loc[0, i, j].astype(np.int), loc[1, i, j].astype(np.int), loc[2, i, j].astype(np.int)]
    slicer[np.isnan(slicer)]=0
    return slicer, sub_loc,loc


if __name__ == '__main__':

    # get data path
    filepath = "datapath/..."
    csvpath = "csvpath/..."
    outputfile = ""

    # get data information in csv
    datainf = pd.read_csv(csvpath, usecols=['Image Data ID', 'oblique coronal angle', 'tail', 'y_head', 'z_head', '{0}'.format(fold_name)])
    namelist = datainf['Image Data ID'].tolist()
    tanthetalist = datainf['oblique coronal angle'].tolist()
    y_taillist = datainf['tail'].tolist()
    y_headlist = datainf['y_head'].tolist()
    z_headlist = datainf['z_head'].tolist()
    splitlist = datainf['train_or_val'].tolist()
    # To take the position of the hippocampal decimals, set the scale to the number of decimals.
    # E.g. 0.1 is to take the slice at the first decimals position.
    scale = 0.1


    # get data name
    datapaths = glob.glob(filepath)
    for datapath in datapaths:
        origin_path = datapath
        # get original data name
        # datapath = datapath.split('_')[-1]
        # datapath = datapath.split('.')[0]

        # get segmentation data name  ps:pve_n = -4; others = -3
        datapath = datapath.split('_')[-3]

        print(datapath)

        for i in namelist:
            if i == datapath:
                index = namelist.index(i)
                split = splitlist[index]

                if split == 'train_set':
                    index = namelist.index(i)
                    tantheta = tanthetalist[index]
                    square_costheta = 1 / (1 + tantheta ** 2)
                    # zn is the tan value of the oblique coronal angle (if you want coronal slice, set zn = 0)
                    zn = int(1000 * tantheta)
                    y_tail = int(y_taillist[index])
                    y_head = int(y_headlist[index])
                    z_head_inversed = int(z_headlist[index])
                    z_head = int(255 * (1 - z_head_inversed / 255))  # the z_head in the table is inversed
                    d = scale * (y_head - y_tail) / square_costheta
                    data = nib.load(origin_path).get_data()  # data type=numpy.memmap


                    xc = data.shape[0]//2
                    zc = 128
                    r = 112  # ImageSize/2
                    noc = [0, 1000, -zn]  # Normal vector of the 2D slice

                    if z_head < zc:
                        y_start = int(y_head + round(tantheta * (zc - z_head)))
                    if z_head > zc:
                        y_start = int(y_head - round(tantheta * (z_head - zc)))
                    if z_head == zc:
                        y_start = y_head

                    # Get an image of a specific location
                    yc = int(round(y_start - d))
                    c = [xc, yc, zc]  # Center of the 2D slice
                    slicer, sub_loc, loc = extract_slice(data, c, noc, r)
                    name_number = str(yc + 1)
                    slice_name = datapath + "_" + name_number + ".png"
                    imageio.imwrite(slice_name, slicer)
                    shutil.move(slice_name, outputfile)

                    # Get multiple consecutive oblique coronal slices
                    # total_slices = 25  # Total number of slices
                    # num_divi = 1
                    # for y in range(y_start - total_slices - 1, y_start - 1):
                    #     if (y % num_divi) == 0:
                    #         c = [xc, y, zc]  # Center of the 2D slice
                    #         slicer, sub_loc, loc = extract_slice(data, c, noc, r)
                    #         name_number = str(y + 1)
                    #         slice_name = datapath + "_" + name_number + ".png"
                    #         imageio.imwrite(slice_name, slicer)
                    #         shutil.move(slice_name, outputfile)
                    #         y += 1

                else:
                    print('pass')



