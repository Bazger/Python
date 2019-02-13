import numpy as np
from ImagePair import ImagePair as Ip
from SingleImage import SingleImage
from Camera import Camera
from matplotlib import pyplot as plt
from MatrixMethods import Compute3DRotationMatrix

import PhotoViewer as pv

class ImageTriple(object):
    def __init__(self, imagePair1, imagePair2):
        """
        Inisialize the ImageTriple class

        :param imagePair1: first image pair
        :param imagePair2: second image pair

        .. warning::

            Check if the relative orientation is solved for each image pair
        """
        self.__imagePair1 = imagePair1
        self.__imagePair2 = imagePair2


    def ComputeScaleBetweenModels(self, cameraPoint1, cameraPoint2, cameraPoint3):
         """
         Compute scale between two models given the relative orientation

         :param cameraPoints1: camera point in first camera space
         :param cameraPoints2: camera point in second camera space
         :param cameraPoints3:  camera point in third camera space

         :type cameraPoints1: np.array 1x3
         :type cameraPoints2: np.array 1x3
         :type cameraPoints3: np.array 1x3


         .. warning::

             This function is empty, need implementation
         """
         if cameraPoint1[0]==-1 or cameraPoint2[0]==-1 or cameraPoint3[0]==-1:
             return
         f = self.__imagePair1.image1.camera.focalLength
         x1 = np.array([[cameraPoint1[0], cameraPoint1[1], -f]]).T
         x2 = np.array([[cameraPoint2[0], cameraPoint2[1], -f]]).T
         x3 = np.array([[cameraPoint3[0], cameraPoint3[1], -f]]).T

         Orient_12 = self.__imagePair1.relativeOrientationImage2
         Orient_23 = self.__imagePair2.relativeOrientationImage2

         b_12 = np.array([[Orient_12[0]], [Orient_12[1]], [Orient_12[2]]])
         b_23 = np.array([[Orient_23[0]], [Orient_23[1]], [Orient_23[2]]])

         R_12 = Compute3DRotationMatrix(Orient_12[3], Orient_12[4], Orient_12[5])
         R_23 = Compute3DRotationMatrix(Orient_23[3], Orient_23[4], Orient_23[5])
         R_13 = np.dot(R_12, R_23)

         V1 = x1
         V2 = np.dot(R_12, x2)
         V3 = np.dot(R_13, x3)

         d1 = np.cross(V1.T, V2.T) / np.linalg.norm(np.cross(V1.T, V2.T))
         d2 = np.cross(V2.T, V3.T) / np.linalg.norm(np.cross(V2.T, V3.T))

         A = np.zeros((6, 6))
         A[:3, 0] = V1[:, 0]
         A[:3, 1] = d1[:, 0]
         A[:3, 3] = -V2[:, 0]
         A[3:, 2] = V2[:, 0]
         A[3:, 3] = d2[:, 0]
         A[3:, 4] = V3[:, 0]
         A[3:, 5] = -b_23[:, 0]


         L = np.zeros((6, 1))
         L[:3, 0] = -b_12[:, 0]
         N = np.dot(A.T, A)
         U = np.dot(A.T, L)
         X = np.dot(np.linalg.inv(N), U)
         S = X[5, 0]
         return S
         pass

    def RayIntersection(self, cameraPoints1, cameraPoints2, cameraPoints3):
        """
        Compute coordinates of the corresponding model point

        :param cameraPoints1: points in camera1 coordinate system
        :param cameraPoints2: points in camera2 coordinate system
        :param cameraPoints3: points in camera3 coordinate system

        :type cameraPoints1 np.array nx3
        :type cameraPoints2: np.array nx3
        :type cameraPoints3: np.array nx3

        :return: point in model coordinate system
        :rtype: np.array nx3

        .. warning::

            This function is empty' need implementation
        """


    def drawModles(self, imagePair1, imagePair2, modelPoints1, modelPoints2):
        """
        Draw two models in the same figure

        :param imagePair1: first image pair
        :param imagePair2:second image pair
        :param modelPoints1: points in the firt model
        :param modelPoints2:points in the second model

        :type modelPoints1: np.array nx3
        :type modelPoints2: np.array nx3

        :return: None

        .. warning::
            This function is empty, need implementation
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        imagePair1.drawImagePair(modelPoints1,ax)
        imagePair2.drawImagePair(modelPoints2,ax)


        plt.show()

if __name__ == '__main__':
    camera = Camera(152, None, None, None, None)
    image1 = SingleImage(camera)
    image2 = SingleImage(camera)
    image3 = SingleImage(camera)
    imagePair1 = ImagePair(image1, image2)
    imagePair2 = ImagePair(image2, image3)
    imageTriple1 = ImageTriple(imagePair11, imagePair22)
