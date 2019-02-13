'''
Created feb 2019
@author:yair,eliezer
@id: 300488939,303062129
'''
import csv
import numpy as np
import PhotoViewer as pv
from Reader import Reader
from Camera import Camera
from SingleImage import SingleImage
from MatrixMethods import PrintMatrix
from ImagePair import ImagePair
from ImageTriple import ImageTriple
from matplotlib import pyplot as plt

if __name__ == '__main__':
    f = 3401.3510535795085
    camFiducial = None
    radialDitortion = np.array([0.04734355,-0.41481274,0,0])
    principlPoint = np.array([2880.959129364667,1854.0884354])
    decentringDistortin = np.array([-0.00146837,-0.00281382,0,0])
    innerOrientationParameters={'a0':principlPoint[0],'a1':1,'a2':0,'b0':principlPoint[1],'b1':0,'b2':-1}
    cam = Camera (f,principlPoint,radialDitortion,decentringDistortin,camFiducial)

    image1 = SingleImage(cam)
    image1.innerOrientationParameters=innerOrientationParameters
    image2 = SingleImage(cam)
    image2.innerOrientationParameters = innerOrientationParameters
    image3 = SingleImage(cam)
    image3.innerOrientationParameters = innerOrientationParameters

    image1_objectPoint = []
    image2_objectPoint = []
    image3_objectPoint = []
    with open('Data/pointModel.txt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            image1_objectPoint.append([float(row[1]), float(row[2])])
            image2_objectPoint.append([float(row[3]), float(row[4])])
            image3_objectPoint.append([float(row[5]), float(row[6])])
    image1_homologPoint = []
    image2_homologPoint = []
    image3_homologPoint = []
    with open('Data/pointHomlog.txt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            image1_homologPoint.append([float(row[1]), float(row[2])])
            image2_homologPoint.append([float(row[3]), float(row[4])])
            image3_homologPoint.append([float(row[5]), float(row[6])])



    image1_objectPoint = image1.ImageToCamera(np.array(image1_objectPoint))
    image2_objectPoint = image2.ImageToCamera(np.array(image2_objectPoint))
    image3_objectPoint = image3.ImageToCamera(np.array(image3_objectPoint))
    image1_homologPoint = image1.ImageToCamera(np.array(image1_homologPoint))
    image2_homologPoint = image2.ImageToCamera(np.array(image2_homologPoint))
    image3_homologPoint = image3.ImageToCamera(np.array(image3_homologPoint))
    imagepair1 = ImagePair(image1, image2)
    x = imagepair1.ComputeDependentRelativeOrientation(image1_homologPoint, image2_homologPoint,np.array([1, 0, 0, 0, 0, 0]))
    imagepair2 = ImagePair(image2, image3)
    y = imagepair2.ComputeDependentRelativeOrientation(image3_homologPoint, image3_homologPoint,np.array([1, 0, 0, 0, 0, 0]))
    x=imagepair1.ImagesToGround(image1_objectPoint,image2_objectPoint,'vector')
    ground=(x['groundpoint'])
    PrintMatrix(ground)
    x = imagepair1.ImagesToGround(image1_homologPoint, image2_homologPoint, 'vector')
    ground2 = (x['groundpoint'])
    PrintMatrix(ground)
    imagepair1.drawImagePair(ground,ground2)



    # x=[]
    #     # y=[]
    #     # z=[]
    #     # for i in ground:
    #     #     if i[0]==-1:
    #     #         continue
    #     #     x.append(i[0])
    #     #     y.append(i[1])
    #     #     z.append(i[2])
    #
    # fig=plt.figure()
    # ax=fig.add_subplot(111, projection='3d')
    # ax.plot(x,y,zs=z,color='black')
    # plt.show()

