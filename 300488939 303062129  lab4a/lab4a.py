'''
Created november 2018
@author:yair,eliezer
@id: 300488939,303062129
'''
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import csv
import numpy as np

from MatrixMethods import PrintMatrix
from SingleImage import SingleImage
from Reader import Reader
from Camera import Camera

if __name__ == '__main__':
    camfile = Reader.ReadCamFile(r'Data\rc30.cam')
    iopfile = Reader.ReadIopFile(r'Data\AR112-3574.iop')
    ipffile = Reader.ReadIpfFile(r'Data\AR112-3574.ipf')
    gpffile = Reader.ReadGpfFile(r'Data\GroundPoints.gpf')[0]
    #print(camfile)
    cam1 = Camera(camfile['f'],[camfile['xp'],camfile['yp']],[camfile['k0'],camfile['k1'],camfile['k2'],camfile['k3']],
            [camfile['p1'],camfile['p2'],camfile['p3'],camfile['p4']],camfile['fiducials'])
    image = SingleImage(cam1)
    x=image.ComputeInnerOrientation(iopfile[0])
    #print(x)
    InnerOrientation = image.ComputeInnerOrientation(iopfile[0][:7:1,:])
   # print(iopfile1)
    # #image.ComputeInnerOrientation(iopfile[0][:-2,:])
    geopar=image.ComputeGeometricParameters()
    #print(geopar)
    inv=image.ComputeInverseInnerOrientation()

    #print(inv)
    #print(ipffile)
    ipffile_1 = np.ones((len(ipffile),2))
    j=0
    for i in ipffile:
        ipffile_1[j, 0] = ipffile[i][0]
        ipffile_1[j, 1] = ipffile[i][1]
        j+=1
    #PrintMatrix(ipffile_1)
    # camerapoint = image.ImageToCamera(ipffile_1)
    # PrintMatrix(camerapoint)
    #print(gpffile)
    gpffile_1 = np.ones((len(gpffile), 3))
    k = 0
    for l in gpffile:
        gpffile_1[k, 0] = gpffile[l][0]
        gpffile_1[k, 1] = gpffile[l][1]
        gpffile_1[k, 2] = gpffile[l][2]
        k =k+ 1



    # for all points
    exteriorOrientation=image.ComputeExteriorOrientation(ipffile_1[:],gpffile_1[:9,],0.1)
    print('all',exteriorOrientation)
    #PrintMatrix(exteriorOrientation['residualsVector'])
    groundpoint=image.ImageToGround_GivenZ(ipffile_1,gpffile_1[:9,2])
    print("A",groundpoint)
    #print(80*'-')

    #for 4 first points and point 6
    ipffile_2=np.ones((6,2))
    ipffile_2[:4, :] = ipffile_1[:4, :]
    ipffile_2[4:,:] = ipffile_1[6]
    gpffile_2=np.ones((6,3))
    gpffile_2[:4, :] = gpffile_1[:4, :]
    gpffile_2[4:,:] = gpffile_1[6]
    exteriorOrientation=image.ComputeExteriorOrientation(ipffile_2,gpffile_2,0.1)
    #print(exteriorOrientation)
    #PrintMatrix(exteriorOrientation['residualsVector'])
    groundpoint=image.ImageToGround_GivenZ(ipffile_1,gpffile_1[:9,2])
    print("B",groundpoint)

    #for 3 first points
    exteriorOrientation=image.ComputeExteriorOrientation(ipffile_1[:3,:],gpffile_1[:3,:],0.1)
    print(exteriorOrientation)
    groundpoint=image.ImageToGround_GivenZ(ipffile_1,gpffile_1[:9,2])
    print("D",groundpoint)

    # for 3 first points and points 7-9
    ipffile_3 = np.ones((6, 2))
    ipffile_3[:3, :] = ipffile_1[:3, :]
    ipffile_3[3:, :] = ipffile_1[6:9,:]
    gpffile_3 = np.ones((6, 3))
    gpffile_3[:3, :] = gpffile_1[:3, :]
    gpffile_3[3:, :] = gpffile_1[6:9,:]
    exteriorOrientation = image.ComputeExteriorOrientation(ipffile_3, gpffile_3, 0.1)
    #print(exteriorOrientation)
    #PrintMatrix(exteriorOrientation['residualsVector'])
    groundpoint = image.ImageToGround_GivenZ(ipffile_1, gpffile_1[:9, 2])
    print("D",groundpoint)