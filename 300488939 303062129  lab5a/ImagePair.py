from Camera import Camera
from SingleImage import SingleImage
from MatrixMethods import Compute3DRotationMatrix, Compute3DRotationDerivativeMatrix, ComputeSkewMatrixFromVector
import numpy as np
from matplotlib import pyplot as plt
import PhotoViewer as pv


class ImagePair(object):

    def __init__(self, image1, image2):
        """
        Initialize the ImagePair class
        :param image1: First image
        :param image2: Second image
        """
        self.__image1 = image1
        self.__image2 = image2
        self.__relativeOrientationImage1 = np.array([0, 0, 0, 0, 0, 0]) # The relative orientation of the first image
        self.__relativeOrientationImage2 = None # The relative orientation of the second image
        self.__absoluteOrientation = None
        self.__isSolved = False # Flag for the relative orientation


    @property
    def isSolved(self):
        """
        Flag for the relative orientation
        returns True if the relative orientation is solved, otherwise it returns False

        :return: boolean, True or False values
        """
        return self.__isSolved

    @property
    def RotationMatrix_Image1(self):
        """
        return the rotation matrix of the first image

        :return: rotation matrix

        :rtype: np.array 3x3
        """
        return Compute3DRotationMatrix(self.__relativeOrientationImage1[0], self.__relativeOrientationImage1[1],
                                       self.__relativeOrientationImage1[2])

    @property
    def RotationMatrix_Image2(self):
        """
        return the rotation matrix of the second image

        :return: rotation matrix

        :rtype: np.array 3x3
        """
        return Compute3DRotationMatrix(self.__relativeOrientationImage2[0], self.__relativeOrientationImage2[1],
                                       self.__relativeOrientationImage2[2])

    @property
    def PerspectiveCenter_Image1(self):
        """
        return the perspective center of the first image

        :return: perspective center

        :rtype: np.array (3, )
        """
        return self.__relativeOrientationImage1[0:3]

    @property
    def PerspectiveCenter_Image2(self):
        """
        return the perspective center of the second image

        :return: perspective center

        :rtype: np.array (3, )
        """
        return self.__relativeOrientationImage2[0:3]

    def drawImagePair(self,modelPoints,homologiPoints):
        '''

        :param modelPoints:
        :return:
        '''

        fig=plt.figure()
        ax=fig.add_subplot(111, projection='3d')
        orient1 = self.__relativeOrientationImage1
        orient2 = self.__relativeOrientationImage2
        R1 = self.RotationMatrix_Image1
        R2 = self.RotationMatrix_Image2
        X0_1 = np.array([[0], [0], [0]])
        X0_2 = np.array([[orient2[0]], [orient2[1]], [orient2[2]]])
        pv.drawRays(modelPoints, X0_1, ax)
        pv.drawRays(modelPoints, X0_2, ax)
        pv.drawRays(homologiPoints, X0_1, ax)
        pv.drawRays(homologiPoints, X0_2, ax)
        f=self.__image1.camera.focalLength
        scale=0.000025
        x = 2*self.__image1.camera.principalPoint[0]
        y = 2*self.__image1.camera.principalPoint[1]
        pv.drawImageFrame(x, y, R1, X0_1, f, scale, ax)
        pv.drawImageFrame(x, y, R2, X0_2, f, scale, ax)
        xm = []
        ym = []
        zm = []
        for i in modelPoints:
            if i[0] == -1:
                continue
            xm.append(i[0])
            ym.append(i[1])
            zm.append(i[2])
        xh = []
        yh = []
        zh = []
        for i in homologiPoints:
            if i[0] == -1:
                continue
            xh.append(i[0])
            yh.append(i[1])
            zh.append(i[2])
            ax.plot(xm, ym, zs=zm, color='blue')
            ax.scatter(xh, yh,zh, c='r', s=50)
        plt.show()

        return
    def ImagesToGround(self, imagePoints1, imagePoints2, Method):
        """
        Computes ground coordinates of homological points

        :param imagePoints1: points in image 1
        :param imagePoints2: corresponding points in image 2
        :param Method: method to use for the ray intersection, three options exist: geometric, vector, Collinearity

        :type imagePoints1: np.array nx2
        :type imagePoints2: np.array nx2
        :type Method: string

        :return: ground points, their accuracies.

        :rtype: dict

        .. warning::

            This function is empty, need implementation


        **Usage example**

        .. code-block:: py

            camera = Camera(152, None, None, None, None)
            image1 = SingleImage(camera)
            image2 = SingleImage(camera)

            imagePoints1 = np.array([[-4.83,7.80],
                                [-4.64, 134.86],
                                [5.39,-100.80],
                                [4.58,55.13],
                                [98.73,9.59],
                                [62.39,128.00],
                                [67.90,143.92],
                                    [56.54,-85.76]])
            imagePoints2 = np.array([[-83.17,6.53],
                                 [-102.32,146.36],
                                 [-62.84,-102.87],
                                 [-97.33,56.40],
                                 [-3.51,14.86],
                                 [-27.44,136.08],
                                 [-23.70,152.90],
                                 [-8.08,-78.07]])

            new = ImagePair(image1, image2)

            new.ImagesToGround(imagePoints1, imagePoints2, 'geometric'))

        """
        exterior1 = self.__relativeOrientationImage1
        exterior2 = self.__relativeOrientationImage2
        camerapoints1 = imagePoints1
        camerapoints2 = imagePoints2
        X01 = exterior1[0]
        Y01 = exterior1[1]
        Z01 = exterior1[2]
        X02 = exterior2[0]
        Y02 = exterior2[1]
        Z02 = exterior2[2]
        f = self.__image1.camera.focalLength
        R1 = self.RotationMatrix_Image1
        R2 = self.RotationMatrix_Image2
        o1 = np.array([[X01], [Y01], [Z01]])
        o2 = np.array([[X02], [Y02], [Z02]])
        groundpoint = np.zeros((len(camerapoints1), 3))
        accuracies = np.zeros((len(camerapoints1), 3))

        for i in range(len(imagePoints1)):
            x1 = np.array([[camerapoints1[i, 0]], [camerapoints1[i, 1]], [-f]])
            x2 = np.array([[camerapoints2[i, 0]], [camerapoints2[i, 1]], [-f]])
            v1 = np.dot(R1, x1)
            v2 = np.dot(R2, x2)
            L = np.array([[X01 - X02], [Y01 - Y02], [Z01 - Z02]])
            A = np.zeros((3, 2))
            A[0, 0] = -v1[0, 0]
            A[1, 0] = -v1[1, 0]
            A[2, 0] = -v1[2, 0]
            A[0, 1] = v2[0, 0]
            A[1, 1] = v2[1, 0]
            A[2, 1] = v2[2, 0]
            N = np.dot(A.T, A)
            U = np.dot(A.T, L)
            X = np.dot(np.linalg.inv(N), U)
            V = np.dot(A, X) - L
            F = o1 + X[0, 0] * v1
            G = o2 + X[1, 0] * v2
            # print(G-F)
            D = np.sqrt((G[0, 0] - F[0, 0]) ** 2 + (G[1, 0] - F[1, 0]) ** 2 + (G[2, 0] - F[2, 0]) ** 2)
            groundpoint[i] = np.array([(F[0, 0] + G[0, 0]) / 2, (F[1, 0] + G[1, 0]) / 2, (F[2, 0] + G[2, 0]) / 2])
            accuracies[i] = np.array([V[0, 0], V[1, 0], V[2, 0]])

        return {'groundpoint': groundpoint, 'accuracies': accuracies}

    def ComputeDependentRelativeOrientation(self, imagePoints1, imagePoints2, initialValues):
        """
         Compute relative orientation parameters

        :param imagePoints1: points in the first image [m"m]
        :param imagePoints2: corresponding points in image 2(homology points) nx2 [m"m]
        :param initialValues: approximate values of relative orientation parameters

        :type imagePoints1: np.array nx2
        :type imagePoints2: np.array nx2
        :type initialValues: np.array (6L,)

        :return: relative orientation parameters.

        :rtype: np.array 5x1 / ADD

        .. warning::

            Can be held either as dictionary or array. For your implementation and decision.

        .. note::

            Do not forget to decide how it is held and document your decision


        **Usage example**

        .. code-block:: py

            camera = Camera(152, None, None, None, None)
            image1 = SingleImage(camera)
            image2 = SingleImage(camera)

            imagePoints1 = np.array([[-4.83,7.80],
                                [-4.64, 134.86],
                                [5.39,-100.80],
                                [4.58,55.13],
                                [98.73,9.59],
                                [62.39,128.00],
                                [67.90,143.92],
                                [56.54,-85.76]])
            imagePoints2 = np.array([[-83.17,6.53],
                                 [-102.32,146.36],
                                 [-62.84,-102.87],
                                 [-97.33,56.40],
                                 [-3.51,14.86],
                                 [-27.44,136.08],
                                 [-23.70,152.90],
                                 [-8.08,-78.07]])
            new = ImagePair(image1, image2)

            new.ComputeDependentRelativeOrientation(imagePoints1, imagePoints2, np.array([1, 0, 0, 0, 0, 0])))

        """
        self.__relativeOrientationImage2 = initialValues[1:]
        f=self.__image1.camera.focalLength
        n = len(imagePoints1)
        imagePoints_1 = np.zeros((n, 3))
        imagePoints_2 = np.zeros((n, 3))

        for i in range(0,n):
            imagePoints_1[i] = [imagePoints1[i, 0], imagePoints1[i, 1], -f]
            imagePoints_2[i] = [imagePoints2[i, 0], imagePoints2[i, 1], -f]

        i = 0
        while i < 20:
            A, B, W = self.Build_A_B_W(imagePoints_1, imagePoints_2, self.__relativeOrientationImage2)
            M = np.dot(B, B.T)
            N = np.dot(np.dot(A.T, np.linalg.inv(M)), A)
            U = np.dot(np.dot(A.T, np.linalg.inv(M)), W)
            DX = -np.dot(np.linalg.inv(N), U)
            self.__relativeOrientationImage2 = self.__relativeOrientationImage2 + DX
            i += 1
        V = -np.dot(np.dot(B.T, np.linalg.inv(M)), np.dot(A, DX) + W)
        sigma =np.sqrt(np.dot(V.T, V) / (imagePoints1.shape[0] - 5))
        var = sigma**2 * np.linalg.inv(N)
        relativeOrientation = self.__relativeOrientationImage2
        initialValues = [1, relativeOrientation[0], relativeOrientation[1], relativeOrientation[2], relativeOrientation[3], relativeOrientation[4]]
        self.__relativeOrientationImage2 = initialValues

        return {'relative Orientation': self.__relativeOrientationImage2, 'accuracies': np.sqrt(np.diag(var))}

    def Build_A_B_W(self, cameraPoints1, cameraPoints2, x):
        """
        Function for computing the A and B matrices and vector w.
        :param cameraPoints1: points in the first camera system
        :param ImagePoints2: corresponding homology points in the second camera system
        :param x: initialValues vector by, bz, omega, phi, kappa ( bx=1)

        :type cameraPoints1: np.array nx3
        :type cameraPoints2: np.array nx3
        :type x: np.array (5,1)

        :return: A ,B matrices, w vector

        :rtype: tuple
        """
        numPnts = cameraPoints1.shape[0] # Number of points

        dbdy = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
        dbdz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])

        dXdx = np.array([1, 0, 0])
        dXdy = np.array([0, 1, 0])


        # Compute rotation matrix and it's derivatives
        rotationMatrix2 = Compute3DRotationMatrix(x[2], x[3], x[4])
        dRdOmega = Compute3DRotationDerivativeMatrix(x[2], x[3], x[4], 'omega')
        dRdPhi = Compute3DRotationDerivativeMatrix(x[2], x[3], x[4], 'phi')
        dRdKappa = Compute3DRotationDerivativeMatrix(x[2], x[3], x[4], 'kappa')

        # Create the skew matrix from the vector [bx, by, bz]
        bMatrix = ComputeSkewMatrixFromVector(np.array([1, x[0], x[1]]))

        # Compute A matrix; the coplanar derivatives with respect to the unknowns by, bz, omega, phi, kappa
        A = np.zeros((numPnts, 5))
        A[:, 0] = np.diag(
            np.dot(cameraPoints1, np.dot(dbdy, np.dot(rotationMatrix2, cameraPoints2.T))))  # derivative in respect to by
        A[:, 1] = np.diag(
            np.dot(cameraPoints1, np.dot(dbdz, np.dot(rotationMatrix2, cameraPoints2.T))))  # derivative in respect to bz
        A[:, 2] = np.diag(
            np.dot(cameraPoints1, np.dot(bMatrix, np.dot(dRdOmega, cameraPoints2.T))))  # derivative in respect to omega
        A[:, 3] = np.diag(
            np.dot(cameraPoints1, np.dot(bMatrix, np.dot(dRdPhi, cameraPoints2.T))))  # derivative in respect to phi
        A[:, 4] = np.diag(
            np.dot(cameraPoints1, np.dot(bMatrix, np.dot(dRdKappa, cameraPoints2.T))))  # derivative in respect to kappa

        # Compute B matrix; the coplanar derivatives in respect to the observations, x', y', x'', y''.
        B = np.zeros((numPnts, 4 * numPnts))
        k = 0
        for i in range(numPnts):
            p1vec = cameraPoints1[i, :]
            p2vec = cameraPoints2[i, :]
            B[i, k] = np.dot(dXdx, np.dot(bMatrix, np.dot(rotationMatrix2, p2vec)))
            B[i, k + 1] = np.dot(dXdy, np.dot(bMatrix, np.dot(rotationMatrix2, p2vec)))
            B[i, k + 2] = np.dot(np.dot(p1vec, np.dot(bMatrix, rotationMatrix2)), dXdx)
            B[i, k + 3] = np.dot(np.dot(p1vec, np.dot(bMatrix, rotationMatrix2)), dXdy)
            k += 4

        # w vector
        w = np.diag(np.dot(cameraPoints1, np.dot(bMatrix, np.dot(rotationMatrix2, cameraPoints2.T))))

        return A, B, w

    def ImagesToModel(self, imagePoints1, imagePoints2, Method):
        """
        Mapping points from image space to model space

        :param imagePoints1: points from the first image
        :param imagePoints2: points from the second image
        :param Method: method for intersection

        :type imagePoints1: np.array nx2
        :type imagePoints2: np.array nx2
        :type Method: string

        :return: corresponding model points
        :rtype: np.array nx3


        .. warning::

            This function is empty, need implementation

        .. note::

            One of the images is a reference, orientation of this image must be set.

        """

    def GroundToImage(self, groundPoints):
        """
        Transforming ground points to image points

        :param groundPoints: ground points [m]

        :type groundPoints: np.array nx3

        :return: corresponding Image points

        :rtype: np.array nx2

        """
        pass  # delete after implementation

    def geometricIntersection(self, cameraPoints1, cameraPoints2):
        """
        Ray Intersection based on geometric calculations.

        :param cameraPoints1: points in the first image
        :param cameraPoints2: corresponding points in the second image

        :type cameraPoints1: np.array nx3
        :type cameraPoints2: np.array nx3

        :return: lambda1, lambda2 scalars

        :rtype: np.array nx2

        .. warning::

            This function is empty, need implementation

        """


    def vectorIntersction(self, cameraPoints1, cameraPoints2):
        """
        Ray Intersection based on vector calculations.

        :param cameraPoints1: points in image space
        :param cameraPoints2: corresponding image points

        :type cameraPoints1: np.array nx
        :type cameraPoints2: np.array nx


        :return: lambda1, lambda2 scalars

        :rtype: np.array nx2

        .. warning::

            This function is empty, need implementation

        """


    def CollinearityIntersection(self, cameraPoints1, cameraPoints2):
        """
        Ray intersection based on the collinearity principle

        :param cameraPoints1: points in image space
        :param cameraPoints2: corresponding image points

        :type cameraPoints1: np.array nx2
        :type cameraPoints2: np.array nx2

        :return: corresponding ground points

        :rtype: np.array nx3

        .. warning::

            This function is empty, need implementation

        """


if __name__ == '__main__':
    camera = Camera(152, None, None, None, None)
    image1 = SingleImage(camera)
    image2 = SingleImage(camera)
    leftCamPnts = np.array([[-4.83,7.80],
                            [-4.64, 134.86],
                            [5.39,-100.80],
                            [4.58,55.13],
                            [98.73,9.59],
                            [62.39,128.00],
                            [67.90,143.92],
                            [56.54,-85.76]])
    rightCamPnts = np.array([[-83.17,6.53],
                             [-102.32,146.36],
                             [-62.84,-102.87],
                             [-97.33,56.40],
                             [-3.51,14.86],
                             [-27.44,136.08],
                             [-23.70,152.90],
                             [-8.08,-78.07]])
    new = ImagePair(image1, image2)

    print(new.ComputeDependentRelativeOrientation(leftCamPnts, rightCamPnts, np.array([1, 0, 0, 0, 0, 0])))