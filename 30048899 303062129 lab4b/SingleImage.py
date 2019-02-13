
import numpy as np
from Camera import Camera
from MatrixMethods import Compute3DRotationMatrix, Compute3DRotationDerivativeMatrix


class SingleImage(object):

    def __init__(self, camera):
        """
        Initialize the SingleImage object

        :param camera: instance of the Camera class
        :param points: points in image space

        :type camera: Camera
        :type points: np.array

        """
        self.__camera = camera
        self.__innerOrientationParameters = None
        self.__isSolved = False
        self.__exteriorOrientationParameters = np.array([0, 0, 0, 0, 0, 0], 'f')
        self.__rotationMatrix = None

    @property
    def innerOrientationParameters(self):
        """
        Inner orientation parameters


        .. warning::

            Can be held either as dictionary or array. For your implementation and decision.

        .. note::

            Do not forget to decide how it is held and document your decision

        :return: inner orinetation parameters

        :rtype: **ADD**
        """


        return self.__innerOrientationParameters

    @property
    def camera(self):
        """
        The camera that took the image

        :rtype: Camera

        """
        return self.__camera

    @property
    def exteriorOrientationParameters(self):
        r"""
        Property for the exterior orientation parameters

        :return: exterior orientation parameters in the following order, **however you can decide how to hold them (dictionary or array)**

        .. math::
            exteriorOrientationParameters = \begin{bmatrix} X_0 \\ Y_0 \\ Z_0 \\ \omega \\ \varphi \\ \kappa \end{bmatrix}

        :rtype: np.ndarray or dict
        """
        return self.__exteriorOrientationParameters

    @exteriorOrientationParameters.setter
    def exteriorOrientationParameters(self, parametersArray):
        r"""

        :param parametersArray: the parameters to update the ``self.__exteriorOrientationParameters``

        **Usage example**

        .. code-block:: py

            self.exteriorOrintationParameters = parametersArray

        """
        self.__exteriorOrientationParameters = parametersArray

    @property
    def rotationMatrix(self):
        """
            The rotation matrix of the image

            Relates to the exterior orientation
           :return: rotation matrix

            :rtype: np.ndarray (3x3)
            """

        from MatrixMethods import Compute3DRotationMatrix
        R = Compute3DRotationMatrix(self.exteriorOrientationParameters['OMEGA'], self.exteriorOrientationParameters['PHI'],
                                    self.exteriorOrientationParameters['KAPPA'])

        return R

    @property
    def isSolved(self):
        """
        True if the exterior orientation is solved

        :return True or False

        :rtype: boolean
        """
        return self.__isSolved

    def ComputeInnerOrientation(self, imagePoints):

        r"""
        Compute inner orientation parameters

        :param imagePoints: coordinates in image space

        :type imagePoints: np.array nx2

        :return: Inner orientation parameters, their accuracies, and the residuals vector

        :rtype: dict

        .. warning::

            This function is empty, need implementation

        .. note::

            - Don't forget to update the ``self.__innerOrinetationParameters`` member. You decide the type
            - The fiducial marks are held within the camera attribute of the object, i.e., ``self.camera.fiducialMarks``
            - return values can be a tuple of dictionaries and arrays.

        **Usage example**

        .. code-block:: py

            fMarks = np.array([[113.010, 113.011],
                       [-112.984, -113.004],
                       [-112.984, 113.004],
                       [113.024, -112.999]])
            img_fmarks = np.array([[-7208.01, 7379.35],
                        [7290.91, -7289.28],
                        [-7291.19, -7208.22],
                        [7375.09, 7293.59]])
            cam = Camera(153.42, np.array([0.015, -0.020]), None, None, fMarks)
            img = SingleImage(camera = cam, points = None)
            inner_parameters, accuracies, residuals = img.ComputeInnerOrientation(img_fmarks)
        """
        L = []
        for i in imagePoints:
            L.append(i[0])
            L.append(i[1])
        L=np.matrix(L).T

        fid=self.camera.fiducialMarks

        A = np.zeros((np.size(L),6))
        e=0
        i, j, k = 3, 4, 5
        for row in A:
            if i==0:
                i, j, k = 3, 4, 5
            else:
                i, j, k = 0, 1, 2
            f=int(e/2)
            A[e,i] = 1
            A[e,j] = fid[f,0]
            A[e,k] = fid[f,1]
            e = e+1

        N = np.dot(A.T,A)
        U = np.dot(A.T,L)
        X = np.dot(np.linalg.inv(N),U)
        V = np.dot(A,X)-L


        n = 16
        u = 6
        sp = np.dot(V.T,V)/(n-u)
        cvar=np.linalg.inv(N)*sp[0,0]
        sv=np.diag(cvar)

        self.__innerOrientationParameters = {'a0':X[0,0],'a1':X[1,0],'a2':X[2,0],'b0':X[3,0],'b1':X[4,0],'b2':X[5,0]}
        dict = {'InnerOrientationParameters': self.__innerOrientationParameters, 'theirAccuracies': V, 'residualsVector': sv}

        return dict

    def ComputeGeometricParameters(self):
        """
        Computes the geometric inner orientation parameters

        :return: geometric inner orientation parameters

        :rtype: dict

        .. warning::

           This function is empty, need implementation

        .. note::

            The algebraic inner orinetation paramters are held in ``self.innerOrientatioParameters`` and their type
            is according to what you decided when initialized them

        """
        tx = self.innerOrientationParameters['a0']
        ty = self.innerOrientationParameters['b0']
        a1 = self.innerOrientationParameters['a1']
        b1 = self.innerOrientationParameters['b1']
        a2 = self.innerOrientationParameters['a2']
        b2 = self.innerOrientationParameters['b2']

        tt = np.arctan(b1/b2)
        gam = np.arctan((a1*np.sin(tt)+a2*np.cos(tt))/(b1*np.sin(tt)+b2*np.cos(tt)))
        sx = a1*np.cos(tt)-a2*np.sin(tt)
        sy = (a1*np.sin(tt)+a2*np.cos(tt))/np.sin(gam)
        gometrik = {'tx':tx,'ty':ty,'tt':tt,'gam':gam,"sx":sx,'sy':sy}

        return  gometrik

    def ComputeInverseInnerOrientation(self):
        """
        Computes the parameters of the inverse inner orientation transformation

        :return: parameters of the inverse transformation

        :rtype: dict

        .. warning::

            This function is empty, need implementation

        .. note::

            The inner orientation algebraic parameters are held in ``self.innerOrientationParameters``
            their type is as you decided when implementing
        """
        a0 = self.innerOrientationParameters['a0']
        a1 = self.innerOrientationParameters['a1']
        a2 = self.innerOrientationParameters['a2']
        b0 = self.innerOrientationParameters['b0']
        b1 = self.innerOrientationParameters['b1']
        b2 = self.innerOrientationParameters['b2']

        mat=np.array([[a1,a2],[b1,b2]])
        mov=np.array([[-a0],[-b0]])
        mat2 = np.linalg.inv(mat)
        mov2=np.dot(mat2,mov)

        InverseInnerOrientation={'a0':mov2[0,0],'a1':mat2[0,0],'a2':mat2[0,1],'b0':mov2[1,0],'b1':mat2[1,0],'b2':mat2[1,1]}
        return InverseInnerOrientation

    def CameraToImage(self, cameraPoints):
        """
        Transforms camera points to image points

        :param cameraPoints: camera points

        :type cameraPoints: np.array nx2

        :return: corresponding Image points

        :rtype: np.array nx2


        .. warning::

            This function is empty, need implementation

        .. note::

            The inner orientation parameters required for this function are held in ``self.innerOrientationParameters``

        **Usage example**

        .. code-block:: py

            fMarks = np.array([[113.010, 113.011],
                       [-112.984, -113.004],
                       [-112.984, 113.004],
                       [113.024, -112.999]])
            img_fmarks = np.array([[-7208.01, 7379.35],
                        [7290.91, -7289.28],
                        [-7291.19, -7208.22],
                        [7375.09, 7293.59]])
            cam = Camera(153.42, np.array([0.015, -0.020]), None, None, fMarks)
            img = SingleImage(camera = cam, points = None)
            img.ComputeInnerOrientation(img_fmarks)
            pts_image = img.Camera2Image(fMarks)

        """
        a0 = self.innerOrientationParameters['a0']
        a1 = self.innerOrientationParameters['a1']
        a2 = self.innerOrientationParameters['a2']
        b0 = self.innerOrientationParameters['b0']
        b1 = self.innerOrientationParameters['b1']
        b2 = self.innerOrientationParameters['b2']
        imagepoint=cameraPoints.copy()
        mat = np.array([[a1, a2], [b1, b2]])
        mov = np.array([[a0], [b0]])
        i=0
        for j in cameraPoints:
            imagepoint[i,0]=a0+a1*j[0]+a2*j[1]
            imagepoint[i, 1] = b0 + b1 * j[0] + b2 * j[1]
            i=i+1
        return imagepoint

    def ImageToCamera(self, imagePoints):
        """

        Transforms image points to ideal camera points

        :param imagePoints: image points

        :type imagePoints: np.array nx2

        :return: corresponding camera points

        :rtype: np.array nx2

        .. warning::

            This function is empty, need implementation

        .. note::

            The inner orientation parameters required for this function are held in ``self.innerOrientationParameters``


        **Usage example**

        .. code-block:: py

            fMarks = np.array([[113.010, 113.011],
                       [-112.984, -113.004],
                       [-112.984, 113.004],
                       [113.024, -112.999]])
            img_fmarks = np.array([[-7208.01, 7379.35],
                        [7290.91, -7289.28],
                        [-7291.19, -7208.22],
                        [7375.09, 7293.59]])
            cam = Camera(153.42, np.array([0.015, -0.020]), None, None, fMarks)
            img = SingleImage(camera = cam, points = None)
            img.ComputeInnerOrientation(img_fmarks)
            pts_camera = img.Image2Camera(img_fmarks)

        """
        a0 = self.ComputeInverseInnerOrientation()['a0']
        a1 = self.ComputeInverseInnerOrientation()['a1']
        a2 = self.ComputeInverseInnerOrientation()['a2']
        b0 = self.ComputeInverseInnerOrientation()['b0']
        b1 = self.ComputeInverseInnerOrientation()['b1']
        b2 = self.ComputeInverseInnerOrientation()['b2']
        cameraPoints= imagePoints.copy()

        i = 0
        for j in imagePoints:
            cameraPoints[i, 0] = a0 + a1 * j[0] + a2 * j[1]
            cameraPoints[i, 1] = b0 + b1 * j[0] + b2 * j[1]
            i = i + 1
        idealcameraPoints=self.camera.CameraToIdealCamera(cameraPoints)
        return idealcameraPoints

    def ComputeExteriorOrientation(self, imagePoints, groundPoints, epsilon):
        """
        Compute exterior orientation parameters.

        This function can be used in conjecture with ``self.__ComputeDesignMatrix(groundPoints)`` and ``self__ComputeObservationVector(imagePoints)``

        :param imagePoints: image points
        :param groundPoints: corresponding ground points

            .. note::

                Angles are given in radians

        :param epsilon: threshold for convergence criteria

        :type imagePoints: np.array nx2
        :type groundPoints: np.array nx3
        :type epsilon: float

        :return: Exterior orientation parameters: (X0, Y0, Z0, omega, phi, kappa), their accuracies, and residuals vector. *The orientation parameters can be either dictionary or array -- to your decision*

        :rtype: dict


        .. warning::

           - This function is empty, need implementation
           - Decide how the parameters are held, don't forget to update documentation

        .. note::

            - Don't forget to update the ``self.exteriorOrientationParameters`` member (every iteration and at the end).
            - Don't forget to call ``cameraPoints = self.ImageToCamera(imagePoints)`` to correct the coordinates              that are sent to ``self.__ComputeApproximateVals(cameraPoints, groundPoints)``
            - return values can be a tuple of dictionaries and arrays.

        **Usage Example**

        .. code-block:: py

            img = SingleImage(camera = cam)
            grdPnts = np.array([[201058.062, 743515.351, 243.987],
                        [201113.400, 743566.374, 252.489],
                        [201112.276, 743599.838, 247.401],
                        [201166.862, 743608.707, 248.259],
                        [201196.752, 743575.451, 247.377]])
            imgPnts3 = np.array([[-98.574, 10.892],
                         [-99.563, -5.458],
                         [-93.286, -10.081],
                         [-99.904, -20.212],
                         [-109.488, -20.183]])
            img.ComputeExteriorOrientation(imgPnts3, grdPnts, 0.3)


        """
        cameraPoints=self.ImageToCamera(imagePoints)
        ex=self.__ComputeApproximateVals(cameraPoints[:],groundPoints[:])
        X0 = self.exteriorOrientationParameters['X0']
        Y0 = self.exteriorOrientationParameters['Y0']
        Z0 = self.exteriorOrientationParameters['Z0']
        KAPPA = self.exteriorOrientationParameters['KAPPA']
        n = len(groundPoints)
        Lb = np.zeros((2 * n, 1))
        i=0
        for j in cameraPoints:
            Lb[i] = j[0]
            Lb[i+1] = j[1]
            i=i+2
        dX=np.ones((6,1))
        while epsilon<dX[0,0] or epsilon<dX[1,0] or  epsilon<dX[2,0]:
            print('+++')
            A = self.__ComputeDesignMatrix(groundPoints[:])
            L0 = self.__ComputeObservationVector(groundPoints[:])
            L = Lb - L0
            N = np.dot(A.T, A)
            U = np.dot(A.T, L)
            dX = np.dot(np.linalg.inv(N), U)
            x0=self.exteriorOrientationParameters['X0'] + dX[0,0]
            y0=self.exteriorOrientationParameters['Y0'] + dX[1,0]
            z0=self.exteriorOrientationParameters['Z0'] + dX[2,0]
            omega = self.exteriorOrientationParameters['OMEGA'] + dX[3, 0]
            phi = self.exteriorOrientationParameters['PHI'] + dX[4, 0]
            kappa=self.exteriorOrientationParameters['KAPPA'] + dX[5,0]
            exteriorOrientationParameters = {'X0': x0, 'Y0': y0, 'Z0': z0, 'KAPPA': kappa, 'OMEGA': omega,'PHI':phi}
            self.exteriorOrientationParameters = exteriorOrientationParameters
            V = np.dot(A,dX) - L
            # if len(L)==6:
            #     break

        sp = np.dot(V.T, V) / np.linalg.matrix_rank(A)
        cvar = np.linalg.inv(N) * sp[0, 0]
        sv = np.diag(cvar)
        return {'exteriorOrientationParameters':self.exteriorOrientationParameters,'theirAccuracies': V, 'residualsVector': sv}


        pass  # delete for implementation

    def GroundToImage(self, groundPoints):
        """
        Transforming ground points to image points

        :param groundPoints: ground points [m]

        :type groundPoints: np.array nx3

        :return: corresponding Image points

        :rtype: np.array nx2

        """

        X0 = self.exteriorOrientationParameters['X0']
        Y0 = self.exteriorOrientationParameters['Y0']
        Z0 = self.exteriorOrientationParameters['Z0']
        R = self.rotationMatrix.T
        f = self.camera.focalLength
        camerapoints = np.ones((len(groundPoints), 2))

        for i in range(len(camerapoints)):
            s = -f/(R[2,0]*(groundPoints[i,0]-X0)+R[2,1]*(groundPoints[i,1]-Y0)+R[2,2]*(groundPoints[i,2]-Z0))
            camerapoints[i, 0] = s*(R[0,0]*(groundPoints[i,0]-X0)+R[0,1]*(groundPoints[i,1]-Y0)+R[0,2]*(groundPoints[i,2]-Z0))
            camerapoints[i, 1] = s*(R[1,0]*(groundPoints[i,0]-X0)+R[1,1]*(groundPoints[i,1]-Y0)+R[1,2]*(groundPoints[i,2]-Z0))
        imagepoint=self.CameraToImage(camerapoints)
        return imagepoint


    def ImageToRay(self, imagePoints):
        """
        Transforms Image point to a Ray in world system

        :param imagePoints: coordinates of an image point

        :type imagePoints: np.array nx2

        :return: Ray direction in world system

        :rtype: np.array nx3

        .. warning::

           This function is empty, need implementation

        .. note::

            The exterior orientation parameters needed here are called by ``self.exteriorOrientationParameters``
        """
        pass  # delete after implementations

    def ImageToGround_GivenZ(self, imagePoints, Z_values):
        """
        Compute corresponding ground point given the height in world system

        :param imagePoints: points in image space
        :param Z_values: height of the ground points


        :type Z_values: np.array nx1
        :type imagePoints: np.array nx2
        :type eop: np.ndarray 6x1

        :return: corresponding ground points

        :rtype: np.ndarray

        .. warning::

             This function is empty, need implementation

        .. note::

            - The exterior orientation parameters needed here are called by ``self.exteriorOrientationParameters``
            - The focal length can be called by ``self.camera.focalLength``

        **Usage Example**

        .. code-block:: py


            imgPnt = np.array([-50., -33.])
            img.ImageToGround_GivenZ(imgPnt, 115.)

        """
        camerapoints=self.ImageToCamera(imagePoints)
        X0 = self.exteriorOrientationParameters['X0']
        Y0 = self.exteriorOrientationParameters['Y0']
        Z0 = self.exteriorOrientationParameters['Z0']
        R=self.rotationMatrix
        f=self.camera.focalLength
        groundpoints=np.ones((len(camerapoints),2))

        for i in range(len(camerapoints)):
            lamda=(Z_values[i]-Z0)/(R[2,0]*camerapoints[i,0]+ R[2,1]*camerapoints[i,1]-R[2,2]*f)
            groundpoints[i,0]=X0+lamda*(R[0,0]*camerapoints[i,0]+ R[0,1]*camerapoints[i,1]-R[0,2]*f)
            groundpoints[i,1]=Y0+lamda*(R[1,0]*camerapoints[i,0]+ R[1,1]*camerapoints[i,1]-R[1,2]*f)

        return groundpoints

    # ---------------------- Private methods ----------------------

    def __ComputeApproximateVals(self, cameraPoints, groundPoints):
        """
        Compute exterior orientation approximate values via 2-D conform transformation

        :param cameraPoints: points in image space (x y)
        :param groundPoints: corresponding points in world system (X, Y, Z)

        :type cameraPoints: np.ndarray [nx2]
        :type groundPoints: np.ndarray [nx3]

        :return: Approximate values of exterior orientation parameters
        :rtype: np.ndarray or dict

        .. note::

            - ImagePoints should be transformed to ideal camera using ``self.ImageToCamera(imagePoints)``. See code below
            - The focal length is stored in ``self.camera.focalLength``
            - Don't forget to update ``self.exteriorOrientationParameters`` in the order defined within the property
            - return values can be a tuple of dictionaries and arrays.

        .. warning::

           - This function is empty, need implementation
           - Decide how the exterior parameters are held, don't forget to update documentation



        """
        n = len(groundPoints)
        L = np.zeros((2*n,1))
        for i in range(0,n,1):
            L[i]=groundPoints[i, 0]
        for i in    range(n,2*n,1):
            L[i] = groundPoints[i-n, 1]

        ZMEMOZA=0
        A = np.zeros((2*n,4))
        for i in range(0,n,1):
            A[i,0] = 1
            A[i,2] = cameraPoints[i,0]
            A[i,3] = cameraPoints[i,1]
            ZMEMOZA += groundPoints[i,2]
        for i in range(n, 2 * n, 1):
            A[i, 1] = 1
            A[i, 2] = -cameraPoints[i-n, 1]
            A[i, 3] = cameraPoints[i-n, 0]
        N = np.dot(A.T, A)
        U = np.dot(A.T, L)
        X = np.dot(np.linalg.inv(N), U)
        lamda = np.sqrt(X[2]**2+X[3]**2)
        K=np.arccos(X[2]/lamda)
        Z0=ZMEMOZA/n+lamda*self.camera.focalLength
        exteriorOrientationParameters={'X0':X[0,0],'Y0':X[1,0],'Z0':Z0[0],'KAPPA':K[0],'OMEGA':0,'PHI':0}
        self.__exteriorOrientationParameters = exteriorOrientationParameters

        # Find approximate values
        return exteriorOrientationParameters# delete when implementing

    def __ComputeObservationVector(self, groundPoints):
        """
            Compute observation vector for solving the exterior orientation parameters of a single image
            based on their approximate values

            :param groundPoints: Ground coordinates of the control points

            :type groundPoints: np.array nx3

            :return: Vector l0

            :rtype: np.array nx1
            """

        n = groundPoints.shape[0]  # number of points

        # Coordinates subtraction
        dX = groundPoints[:, 0] - self.exteriorOrientationParameters['X0']
        dY = groundPoints[:, 1] - self.exteriorOrientationParameters['Y0']
        dZ = groundPoints[:, 2] - self.exteriorOrientationParameters['Z0']
        dXYZ = np.vstack([dX, dY, dZ])
        rotated_XYZ = np.dot(self.rotationMatrix.T, dXYZ).T

        l0 = np.empty(n * 2)

        # Computation of the observation vector based on approximate exterior orientation parameters:
        l0[::2] = -self.camera.focalLength * rotated_XYZ[:, 0] / rotated_XYZ[:, 2]
        l0[1::2] = -self.camera.focalLength * rotated_XYZ[:, 1] / rotated_XYZ[:, 2]

        l0 = np.reshape(l0, (len(l0), 1))
        return l0


    def __ComputeDesignMatrix(self, groundPoints):
        """
            Compute the derivatives of the collinear law (design matrix)

            :param groundPoints: Ground coordinates of the control points

            :type groundPoints: np.array nx3

            :return: The design matrix

            :rtype: np.array nx6

            """

        # initialization for readability
        omega = self.exteriorOrientationParameters['OMEGA']
        phi = self.exteriorOrientationParameters['PHI']
        kappa = self.exteriorOrientationParameters['KAPPA']

        # Coordinates subtraction
        dX = groundPoints[:, 0] - self.exteriorOrientationParameters['X0']
        dY = groundPoints[:, 1] - self.exteriorOrientationParameters['Y0']
        dZ = groundPoints[:, 2] - self.exteriorOrientationParameters['Z0']
        dXYZ = np.vstack([dX, dY, dZ])

        rotationMatrixT = self.rotationMatrix.T
        rotatedG = rotationMatrixT.dot(dXYZ)
        rT1g = rotatedG[0, :]
        rT2g = rotatedG[1, :]
        rT3g = rotatedG[2, :]

        focalBySqauredRT3g = self.camera.focalLength / rT3g ** 2

        dxdg = rotationMatrixT[0, :][None, :] * rT3g[:, None] - rT1g[:, None] * rotationMatrixT[2, :][None, :]
        dydg = rotationMatrixT[1, :][None, :] * rT3g[:, None] - rT2g[:, None] * rotationMatrixT[2, :][None, :]

        dgdX0 = np.array([-1, 0, 0], 'f')
        dgdY0 = np.array([0, -1, 0], 'f')
        dgdZ0 = np.array([0, 0, -1], 'f')

        # Derivatives with respect to X0
        dxdX0 = -focalBySqauredRT3g * np.dot(dxdg, dgdX0)
        dydX0 = -focalBySqauredRT3g * np.dot(dydg, dgdX0)

        # Derivatives with respect to Y0
        dxdY0 = -focalBySqauredRT3g * np.dot(dxdg, dgdY0)
        dydY0 = -focalBySqauredRT3g * np.dot(dydg, dgdY0)

        # Derivatives with respect to Z0
        dxdZ0 = -focalBySqauredRT3g * np.dot(dxdg, dgdZ0)
        dydZ0 = -focalBySqauredRT3g * np.dot(dydg, dgdZ0)

        dRTdOmega = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'omega').T
        dRTdPhi = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'phi').T
        dRTdKappa = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'kappa').T

        gRT3g = dXYZ * rT3g

        # Derivatives with respect to Omega
        dxdOmega = -focalBySqauredRT3g * (dRTdOmega[0, :][None, :].dot(gRT3g) -
                                          rT1g * (dRTdOmega[2, :][None, :].dot(dXYZ)))[0]

        dydOmega = -focalBySqauredRT3g * (dRTdOmega[1, :][None, :].dot(gRT3g) -
                                          rT2g * (dRTdOmega[2, :][None, :].dot(dXYZ)))[0]

        # Derivatives with respect to Phi
        dxdPhi = -focalBySqauredRT3g * (dRTdPhi[0, :][None, :].dot(gRT3g) -
                                        rT1g * (dRTdPhi[2, :][None, :].dot(dXYZ)))[0]

        dydPhi = -focalBySqauredRT3g * (dRTdPhi[1, :][None, :].dot(gRT3g) -
                                        rT2g * (dRTdPhi[2, :][None, :].dot(dXYZ)))[0]

        # Derivatives with respect to Kappa
        dxdKappa = -focalBySqauredRT3g * (dRTdKappa[0, :][None, :].dot(gRT3g) -
                                          rT1g * (dRTdKappa[2, :][None, :].dot(dXYZ)))[0]

        dydKappa = -focalBySqauredRT3g * (dRTdKappa[1, :][None, :].dot(gRT3g) -
                                          rT2g * (dRTdKappa[2, :][None, :].dot(dXYZ)))[0]

        # all derivatives of x and y
        dd = np.array([np.vstack([dxdX0, dxdY0, dxdZ0, dxdOmega, dxdPhi, dxdKappa]).T,
                       np.vstack([dydX0, dydY0, dydZ0, dydOmega, dydPhi, dydKappa]).T])

        a = np.zeros((2 * dd[0].shape[0], 6))
        a[0::2] = dd[0]
        a[1::2] = dd[1]


        return a


if __name__ == '__main__':
    fMarks = np.array([[113.010, 113.011],
                       [-112.984, -113.004],
                       [-112.984, 113.004],
                       [113.024, -112.999]])
    img_fmarks = np.array([[-7208.01, 7379.35],
                           [7290.91, -7289.28],
                           [-7291.19, -7208.22],
                           [7375.09, 7293.59]])
    cam = Camera(153.42, np.array([0.015, -0.020]), None, None, fMarks)
    img = SingleImage(camera = cam)
    print(img.ComputeInnerOrientation(img_fmarks))

    print(img.ImageToCamera(img_fmarks))

    print(img.CameraToImage(fMarks))

    GrdPnts = np.array([[5100.00, 9800.00, 100.00]])
    print(img.GroundToImage(GrdPnts))

    imgPnt = np.array([23.00, 25.00])
    print(img.ImageToRay(imgPnt))

    imgPnt2 = np.array([-50., -33.])
    print(img.ImageToGround_GivenZ(imgPnt2, 115.))

    # grdPnts = np.array([[201058.062, 743515.351, 243.987],
    #                     [201113.400, 743566.374, 252.489],
    #                     [201112.276, 743599.838, 247.401],
    #                     [201166.862, 743608.707, 248.259],
    #                     [201196.752, 743575.451, 247.377]])
    #
    # imgPnts3 = np.array([[-98.574, 10.892],
    #                      [-99.563, -5.458],
    #                      [-93.286, -10.081],
    #                      [-99.904, -20.212],
    #                      [-109.488, -20.183]])
    #
    # intVal = np.array([200786.686, 743884.889, 954.787, 0, 0, 133 * np.pi / 180])
    #
    # print img.ComputeExteriorOrientation(imgPnts3, grdPnts, intVal)
