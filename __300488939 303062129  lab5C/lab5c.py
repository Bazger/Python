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
    with open('Data/pointHomlog2.txt') as csvfile:
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
    o2 = imagepair1.ComputeDependentRelativeOrientation(image1_homologPoint, image2_homologPoint,np.array([1, 0, 0, 0, 0, 0]))

    imagepair2 = ImagePair(image2, image3)
    o3 = imagepair2.ComputeDependentRelativeOrientation(image3_homologPoint, image3_homologPoint,np.array([1, 0, 0, 0, 0, 0]))

    #x1=imagepair1.ImagesToGround(image1_homologPoint ,image2_homologPoint,'vector')
    #ground=(x1['groundpoint'])
    # PrintMatrix(ground)
   # x2 = imagepair2.ImagesToGround(image2_homologPoint, image3_homologPoint, 'vector')
    #ground2 = (x2['groundpoint'])
    # PrintMatrix(ground2)
    x1 = imagepair1.ImagesToGround(image1_objectPoint, image2_objectPoint, 'vector')
    ground = (x1['groundpoint'])
    # PrintMatrix(ground)
    x2 = imagepair2.ImagesToGround(image2_objectPoint , image3_objectPoint, 'vector')
    ground2 = (x2['groundpoint'])
    # PrintMatrix(ground2)
    # imagepair1.drawImagePair(ground,ground2)



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
    R2=imagepair1.RotationMatrix_Image2
    R3= imagepair2.RotationMatrix_Image2
    O2 = np.reshape(o2['relative Orientation'],(6,1))[:3,:]
    O3 = np.reshape(o3['relative Orientation'],(6,1))[:3,:]
    # print(O2,O3)
    O3=O2+np.dot(R2,O3)
    # print(O3)
    R3=np.dot(R2,R3)
    # print(R3)
    phi=np.arcsin(R3[0,2])
    omega=np.arcsin(-R3[1,2]/np.cos(phi))
    kappa=np.arccos(R3[0,0]/np.cos(phi))
    # print(omega,phi ,kappa)
    temp=np.array([O3[0][0],O3[1][0],O3[2][0],omega,phi,kappa])
    # print(o2)
    # o3 = imagepair2.ComputeDependentRelativeOrientation(image3_homologPoint, image3_homologPoint, temp)
    imagepair2.relativeOrientationImage1=(temp)
    imageTriple=ImageTriple(imagepair1,imagepair2)
    # imageTriple.drawModles(imagepair1,imagepair2,ground,ground2)
    kanam=[]
    for i in range(0,len(image1_objectPoint)):
        kanam.append(imageTriple.ComputeScaleBetweenModels(image1_objectPoint[i],image2_objectPoint[i],image3_objectPoint[i]))
    print(np.average(kanam))
    print(np.std(kanam))
    R2 = imagepair1.RotationMatrix_Image2
    R3 = imagepair2.RotationMatrix_Image2
    O2 = np.reshape(o2['relative Orientation'], (6, 1))[:3, :]
    O3 = np.reshape(o3['relative Orientation'], (6, 1))[:3, :]*(1+np.average(kanam))
    # print(O2,O3)
    O3 = O2 + np.dot(R2, O3)
    print(O3)
    R3 = np.dot(R2, R3)
    # print(R3)
    phi = np.arcsin(R3[0, 2])
    omega = np.arcsin(-R3[1, 2] / np.cos(phi))
    kappa = np.arccos(R3[0, 0] / np.cos(phi))
    groundP=(ground+ground2)/2
    # PrintMatrix((ground+ground2)/2)
    # imageTriple.drawModles(imagepair1, imagepair2, ground2, groundP)
"""מעבדה 5ג"""
x=groundP[1]-groundP[2]
v1=groundP[1]-groundP[3]
z=np.cross(x,v1)
print(z)

R1=imagepair1.RotationLevelModel(('x',x), ('z',z))
print(R1)

v2=groundP[1]-groundP[4]
z=np.cross(x,v2)
R2=imagepair1.RotationLevelModel(('x',x), ('z',z))
print(80*'-')
print(R2)
print(80*"-")
print(groundP)

""""
dist2211=np.linalg.norm(groundP[-1]-groundP[3])
dist1611=np.linalg.norm(groundP[-3]-groundP[3])
dist15=np.linalg.norm(groundP[0]-groundP[2])
dist1317=np.linalg.norm(groundP[-2]-groundP[4])
dist175=np.linalg.norm(groundP[-2]-groundP[2])
"""
dist111=np.linalg.norm(groundP[-4]-groundP[0])
dist39=np.linalg.norm(groundP[2]-groundP[6])
dist13=np.linalg.norm(groundP[0]-groundP[2])
dist1113=np.linalg.norm(groundP[-2]-groundP[-4])
dist1112=np.linalg.norm(groundP[-4]-groundP[-3])

dists=np.array([dist111,dist39,dist13,dist13,dist1112])
print('מרחקים במערכת המודל:')
print(dists)
distsWorld=np.array([13.1,19.3,19.3,19.3,18.6])

scales=np.zeros(distsWorld.size)
sum=0
for i in range(distsWorld.size):
    scales[i]=distsWorld[i]/dists[i]
    sum=sum+scales[i]

print('קנ"מ מחושב:')
print(scales)
print('ממוצע קנ"מ')
sumScale=sum/scales.size
print(sumScale)
print('סטיית התקן:')
print(np.std(scales))

sizeAfterScale=dists*sumScale
print('גודל מחושב לאחר קנ"מ:')
print(sizeAfterScale)
print('גודל מדוד במטרים:')
print(distsWorld)
imageTriple.drawModles(imagepair1, imagepair2, ground2*sumScale, groundP*sumScale)

