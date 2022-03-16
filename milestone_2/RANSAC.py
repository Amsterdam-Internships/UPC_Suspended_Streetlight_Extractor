import random

import numpy as np

class Line:
    """
    Based on:
    Implementation for 3D Line RANSAC.
    This object finds the equation of a line in 3D space using RANSAC method.
    This method uses 2 points from 3D space and computes a line. The selected candidate will be the line with more inliers inside the radius theshold.
    ![3D line](https://raw.githubusercontent.com/leomariga/pyRANSAC-3D/master/doc/line.gif "3D line")
    ---
    """

    def __init__(self):
        self.inliers = []
        self.model_coef = []
        self.pt_samples = []

    def coef_matrix(self, matrix):
        assert matrix.shape[1] == 3
        X = matrix[:,0]
        Y = matrix[:,1]
        return np.c_[X, Y, X*Y, X*X, Y*Y, np.ones(matrix.shape[0])]

    def fit(self, pts, residual_threshold=0.2, max_trials=1000):
        """
        Find the best equation for the 3D line. The line in a 3d enviroment is defined as y = Ax+B, but A and B are vectors intead of scalars.
        :param pts: 3D point cloud as a `np.array (N,3)`.
        :param residual_threshold: Threshold distance from the line which is considered inlier.
        :param max_trials: Number of maximum iteration which RANSAC will loop over.
        :returns:
        - `A`: 3D slope of the line (angle) `np.array (1, 3)`
        - `B`: Axis interception as `np.array (1, 3)`
        - `inliers`: Inlier's index from the original point cloud. `np.array (1, M)`
        ---
        """
        n_points = pts.shape[0]
        best_inliers = []

        for it in range(max_trials):

            # Samples 3 random points
            id_samples = random.sample(range(0, n_points), 3)
            pt_samples = pts[id_samples]

            # The plane defined by three points
            A = self.coef_matrix(pt_samples)
            C = np.linalg.lstsq(A, pt_samples[:,2])[0]

            # Distance from a point to fitted plane
            pt_id_inliers = []  # list of inliers ids
            Z_points = np.dot(self.coef_matrix(pts), C)
            dist_pt = Z_points - pts[:,2]

            # Select indexes where distance is biggers than the threshold
            pt_id_inliers = np.where(np.abs(dist_pt) <= residual_threshold)[0]

            if len(pt_id_inliers) > len(best_inliers):
                best_inliers = pt_id_inliers
                self.inliers = np.zeros(n_points, dtype=bool)
                self.inliers[best_inliers] = True
                self.model_coef = C
                self.pt_samples = pt_samples

        return self.model_coef, self.inliers, self.pt_samples

    def fit_gap(self, pts_a, pts_b, pts_gap, residual_threshold=0.2):
        """
        Find the best equation for the 3D line. The line in a 3d enviroment is defined as y = Ax+B, but A and B are vectors intead of scalars.
        :param pts: 3D point cloud as a `np.array (N,3)`.
        :param residual_threshold: Threshold distance from the line which is considered inlier.
        :param max_trials: Number of maximum iteration which RANSAC will loop over.
        :returns:
        - `A`: 3D slope of the line (angle) `np.array (1, 3)`
        - `B`: Axis interception as `np.array (1, 3)`
        - `inliers`: Inlier's index from the original point cloud. `np.array (1, M)`
        ---
        """
        n_points = pts_gap.shape[0]
        best_inliers = []
        pt_samples = np.vstack([pts_a,pts_b])

        # The plane defined by three points
        A = self.coef_matrix(pt_samples)
        C = np.linalg.lstsq(A, pt_samples[:,2])[0]

        # Distance from a point to fitted plane
        pt_id_inliers = []  # list of inliers ids
        Z_points = np.dot(self.coef_matrix(pts_gap), C)
        dist_pt = Z_points - pts_gap[:,2]

        # Select indexes where distance is biggers than the threshold
        pt_id_inliers = np.where(np.abs(dist_pt) <= residual_threshold)[0]

        best_inliers = pt_id_inliers
        self.inliers = np.zeros(n_points, dtype=bool)
        self.inliers[best_inliers] = True
        self.model_coef = C
        self.pt_samples = pt_samples

        return self.model_coef, self.inliers, self.pt_samples

    def inlier_outlier(self, pts, residual_threshold):

        n_points = pts.shape[0]
        Z_points = np.dot(self.coef_matrix(pts), self.model_coef)
        dist_pt = Z_points - pts[:,2]

        # Select indexes where distance is biggers than the threshold
        pt_id_inliers = np.where(np.abs(dist_pt) <= residual_threshold)[0]

        inliers = np.zeros(n_points, dtype=bool)
        inliers[pt_id_inliers] = True

        return inliers