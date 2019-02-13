
import numpy as np

class Camera(object):

    def __init__(self, focal_length, principal_point, radial_distortions, decentering_distortions, fiducial_marks):
        """
        Initialize the Camera object

        :param focal_length: focal length of the camera(mm)
        :param principal_point: principle point
        :param radial_distortions: the radial distortion parameters K0, K1, K2
        :param decentering_distortions: decentering distortion parameters P0, P1, P2
        :param fiducial_marks: fiducial marks in camera space

        :type focal_length: double
        :type principal_point: np.array
        :type radial_distortions: np.array
        :type decentering_distortions: np.array
        :type fiducial_marks: np.array

        """
        # private parameters
        self.__focal_length = focal_length
        self.__principal_point = principal_point
        self.__radial_distortions = radial_distortions
        self.__decentering_distortions = decentering_distortions
        self.__fiducial_marks = fiducial_marks
        self.__CalibrationParam = None

    @property
    def focalLength(self):
        """
        Focal length of the camera

        :return: focal length

        :rtype: float

        """
        return self.__focal_length

    @focalLength.setter
    def focalLength(self, val):
        """
        Set the focal length value

        :param val: value for setting

        :type: float

        """

        self.__focal_length = val

    @property
    def fiducialMarks(self):
        """
        Fiducial marks of the camera, by order

        :return: fiducial marks of the camera

        :rtype: np.array nx2

        """

        return self.__fiducial_marks

    @property
    def principalPoint(self):
        """
        Principal point of the camera

        :return: principal point coordinates

        :rtype: np.ndarray

        """

        return self.__principal_point

    def CameraToIdealCamera(self, camera_points):
        """
        Transform camera coordinates to an ideal system.

        :param camera_points: set of points in camera space

        :type camera_points: np.array nx2

        :return: fixed point set

        :rtype: np.array nx2

        .. warning::

            This function is empty, need implementation
        """
        idealCam = camera_points.copy()
        for i in idealCam:
            if i[0]==-1:
                continue
            xp,yp=self.CorrectionToPrincipalPoint(i)
            dx,dy=self.ComputeDecenteringDistortions(i)
            deltax,deltay=self.ComputeRadialDistortions(i)
            i[0] = i[0] - xp + dx + deltax
            i[1] = i[1] - yp + dy + deltay
        return idealCam

    def IdealCameraToCamera(self, camera_points):
        r"""
        Transform from ideal camera to camera with distortions

        :param camera_points: points in ideal camera space

        :type camera_points: np.array nx2

        :return: corresponding points in image space

        :rtype: np.array nx2

        .. warning::

            This function is empty, need implementation
        """
        idealCam=camera_points.copy()
        for i in idealCam:
            xp, yp = self.CorrectionToPrincipalPoint(i)
            dx, dy = self.ComputeDecenteringDistortions(i)
            deltax, deltay = self.ComputeRadialDistortions(i)
            i[0] = i[0] + xp - dx - deltax
            i[1] = i[1] + yp - dy - deltay
        return idealCam

    def ComputeDecenteringDistortions(self, camera_points):
        """
        Compute decentering distortions for given points

        :param camera_points: points in camera space

        :type camera_points: np.array nx2

        :return: decentering distortions: d_x, d_y

        :rtype: tuple of np.array

        .. warning::

            This function is empty, need implementation
        """

        p1 = self.__decentering_distortions[0]
        p2 = self.__decentering_distortions[1]
        p3 = self.__decentering_distortions[2]
        p4 = self.__decentering_distortions[3]
        xp = self.__principal_point[0]*0.0024
        yp = self.__principal_point[1]*0.0024
        x = camera_points[0]*0.0024
        y = camera_points[1]*0.0024
        xtag = x - xp
        ytag = y - yp
        r = np.sqrt((xtag)**2 + (ytag)**2)
        dx = (1 + p3 * r ** 2 + p4 * r ** 4) * (p1 * (r ** 2 + 2 *xtag ** 2) + 2 * p2 * xtag * ytag)
        dy = (1 + p3 * r ** 2 + p4 * r ** 4) * (p2 * (r ** 2 + 2 * ytag ** 2) + 2 * p1 * xtag * ytag)
        return dx,dy

    def ComputeRadialDistortions(self, camera_points):
        """
        Compute radial distortions for given points

        :param camera_points: points in camera space

        :type camera_points: np.array nx2

        :return: radial distortions: delta_x, delta_y

        :rtype: tuple of np.array

        """
        k0 = self.__radial_distortions[0]
        k1 = self.__radial_distortions[1]
        k2 = self.__radial_distortions[2]
        k3 = self.__radial_distortions[3]
        xp = self.__principal_point[0]*0.0024
        yp = self.__principal_point[1]*0.0024
        x = camera_points[0]*0.0024
        y = camera_points[1]*0.0024
        r = np.sqrt(x**2+y**2)
        xx = (x-xp)*(k0 + k1*r**2 + k2*r**4 + k3*r**6 )
        yy = (y-yp)*(k0 + k1*r**2 + k2*r**4 + k3*r**6 )

        return xx, yy

    def CorrectionToPrincipalPoint(self, camera_points):
        """
        Correction to principal point

        :param camera_points: sampled image points

        :type: np.array nx2

        :return: corrected image points

        :rtype: np.array nx2

        .. warning::

            This function is empty, need implementation

        .. note::

            The principal point is an attribute of the camera object, i.e., ``self.principalPoint``


        """
        # x = camera_points[0]
        # y = camera_points[1]
        xp = self.__principal_point[0]
        yp = self.__principal_point[1]
        # xtag = x - xp
        # ytag = y - yp
        return 0,0



if __name__ == '__main__':

    f0 = 4360.
    xp0 = 2144.5
    yp0 = 1424.5
    K1 = 0
    K2 = 0
    P1 = 0
    P2 = 0

    # define the initial values vector
    cam = Camera(f0, np.array([xp0, yp0]), np.array([K1, K2]),np.array([P1, P2]), None)
