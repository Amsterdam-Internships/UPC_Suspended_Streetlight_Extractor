"""Cable Extractor"""

import set_path
import numpy as np
import logging

from src.abstract_processor import AbstractProcessor
from src.utils.clip_utils import poly_clip, poly_box_clip
from milestone_2.cloud_utils import neighborhood_pca, main_direction, unit_vector, angle_between

from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
from shapely.geometry import LineString
from sklearn.cluster import DBSCAN
import pyransac3d as pyrsc
import open3d as o3d

logger = logging.getLogger(__name__)


class CableExtractor(AbstractProcessor):
    """
    Parameters
    ----------
    label : int
        Class label to use for this fuser.
    ahn_reader : AHNReader object
        Elevation data reader.
    bgt_reader : BGTPolyReader object
        Used to load road part polygons.
    """

    def __init__(self, label, neighborhood_method='radius', min_points=4,
                grow_params={'grow_length':1, 'section_length':3, 'cable_width_buffer':0.1},
                merge_params={'max_angle_A':45, 'max_dist_A':.5, 'max_angle_B':5, 'max_dist_B':5}):
        super().__init__(label)
        self.neighborhood_method = neighborhood_method
        self.min_points = min_points
        self.grow_params = grow_params
        self.merge_params = merge_params

    def _get_linear_points(self, points, method, eps_1=0.001, eps_2=0.1):
        """Based on PCA identify linear points. """

        linearity_mask = np.zeros(len(points), dtype=bool)

        eig_val_vec = neighborhood_pca(points, method)
        assert len(eig_val_vec) == len(points)

        # TODO: 
        # + New linearty formula
        # + Other geometric shapes (Planar & Volumetric)
        for i in range(len(eig_val_vec)):
            eig_val = eig_val_vec[i][0]
            if eig_val is not None:

                # Sort eigenvalues λ1 > λ2 > λ3
                idx = eig_val.argsort()[::-1]   
                eig_val = eig_val[idx]

                # Linearity Check
                cond_1 = (abs(eig_val[1]-eig_val[2]) < eps_1)
                cond_2 = (eig_val[0] > eig_val[1] + eps_2)
                if cond_1 and cond_2:
                    linearity_mask[i] = True

        return linearity_mask

    def _get_cable_clusters(self, points, mask, min_points):

        # Cluster the potential seed points.
        clustering = (DBSCAN(eps=.75, min_samples=4, p=2).fit(points[mask]))
        cluster_labels, counts = np.unique(clustering.labels_, return_counts=True) # Get cluster labels and sizes.
        
        if min_points > 1:
            # Only keep clusters with size at least min_points.
            cluster_labels = cluster_labels[counts >= min_points]

        # Create cable cluster labels
        clustering_labels_ = np.full(len(points), -1)
        mask_ids = np.where(mask)[0]
        cluster_counts = 0
        for cl in set(cluster_labels).difference((-1,)):
            cluster_counts += 1
            c_mask = clustering.labels_ == cl
            clustering_labels_[mask_ids[c_mask]] = cluster_counts

        return clustering_labels_ 

    def _grow_section(self, section_points, grow_points, res_threshold=0.1):

        inlier_mask = np.zeros(len(grow_points), dtype=bool)

        # least-square curve fit
        X_in = section_points[:, 0]
        A = np.c_[X_in, X_in*X_in, np.ones(len(X_in))]
        C = np.linalg.lstsq(A, section_points[:, 1], rcond=None)[0]

        # Distance from unlabeled points to fitted curve
        X_out = grow_points[:, 0]
        Z_points = np.dot(np.c_[X_out, X_out*X_out, np.ones(len(X_out))], C)
        residuals = Z_points - grow_points[:, 1]
        inlier_mask = np.abs(residuals) < res_threshold

        return inlier_mask

    def _grow_clusters(self, points, cluster_labels, grow_length, section_length, cable_width_buffer):
        """
        Grows linear clusters in both directions based on Least Square Fit.

        Parameters:
        --------------
        `grow_length`: float
            grow limit in meters.
        `section_length`: float
            length of cluster to use in Least Square Fit.
        `cable_width_buffer`: float
            width of the cable-cut.

        Returns: labels
        """
        labels = cluster_labels.copy()
        #labels_mask = np.zeros(len(labels), dtype=bool)

        cl_labels = np.unique(labels)

        for cl in set(cl_labels).difference((-1,)):
            cl_mask = labels == cl
            cl_points = points[cl_mask]

            # Create directional Axis
            cl_dir = main_direction(cl_points[:,:2])
            dXY_v = unit_vector(cl_dir) * grow_length
            cl_dir_axis = np.dot(cl_points[:,:2], cl_dir)
            cl_points_dir = np.vstack([cl_dir_axis, cl_points[:,2]]).T
            
            # if length > 1 meter
            cluster_length = cl_dir_axis.max() - cl_dir_axis.min()
            if cluster_length > 1:

                # grow head-section
                head_XY = cl_points[cl_dir_axis.argmax(),:2]
                grow_area = LineString([head_XY, head_XY + dXY_v]).buffer(cable_width_buffer)
                grow_mask = poly_clip(points[labels == -1], grow_area)
                grow_dir_axis = np.dot(points[labels == -1][grow_mask,:2], cl_dir)
                grow_points_dir = np.vstack([grow_dir_axis, points[labels == -1][grow_mask,2]]).T
                head_points_dir = cl_points_dir[cl_dir_axis > cl_dir_axis.max() - section_length]
                inlier_mask = self._grow_section(head_points_dir, grow_points_dir)
                grow_ids = np.where(labels == -1)[0][grow_mask]
                labels[grow_ids[inlier_mask]] = cl
                #labels_mask[grow_ids[inlier_mask]] = True

                # Grow tail-section
                tail_XY = cl_points[cl_dir_axis.argmin(),:2]
                grow_area = LineString([tail_XY, tail_XY - dXY_v]).buffer(cable_width_buffer)
                grow_mask = poly_clip(points[labels == -1], grow_area)
                grow_dir_axis = np.dot(points[labels == -1][grow_mask,:2], cl_dir)
                grow_points_dir = np.vstack([grow_dir_axis, points[labels == -1][grow_mask,2]]).T
                tail_points_dir = cl_points_dir[cl_dir_axis < cl_dir_axis.min() + section_length]
                inlier_mask = self._grow_section(tail_points_dir, grow_points_dir)
                grow_ids = np.where(labels == -1)[0][grow_mask]
                labels[grow_ids[inlier_mask]] = cl

        return labels#, label_mask

    def _nearest_points(self, pts_a, pts_b):
        dist = cdist(pts_a, pts_b)
        idx_a, idx_b = np.unravel_index(np.argmin(dist), dist.shape)
        return pts_a[idx_a], pts_b[idx_b], dist.min()

    def _catenary_merge(self, points, mask_A, mask_B, mask_unlabelled):

        # Define points
        pts_A = points[mask_A]
        pts_B = points[mask_B]
        pts = np.vstack([pts_A, pts_B])
        cable_dir = main_direction(pts[:,:2])
        
        # Define search gap
        pA, pB, _ = self._nearest_points(pts_A, pts_B)
        merge_line = LineString([pA, pB])
        fit_mask = poly_clip(pts, merge_line.buffer(2.5))

        # Trim cable points
        d_pts = np.dot(pts[fit_mask][:,:2], cable_dir)
        d_shift = d_pts.min()
        d_pts -= d_shift
        z_pts = pts[fit_mask][:,2]

        # Gap points
        gap_mask = poly_box_clip(points[mask_unlabelled], merge_line.buffer(.1), bottom=z_pts.min(), top=z_pts.max())
        gap_pts = points[mask_unlabelled][gap_mask]
        d_gap_pts = np.dot(gap_pts[:,:2], cable_dir) - d_shift
        z_gap_pts = gap_pts[:,2]

        # Curve fit on cable segments
        popt, _ = curve_fit(catenary_func, d_pts, z_pts)

        # Evaluate fit on cable
        dist_z = abs(catenary_func(d_pts, *popt) - z_pts)
        fit_inliers = dist_z < 0.1
        fit_score = np.sum(fit_inliers) / len(fit_inliers)

        # Evaluate fit on gap
        dist_gap_z = abs(catenary_func(d_gap_pts, *popt) - z_gap_pts)
        gap_inliers = dist_gap_z < 0.1
        gap_score = np.sum(gap_inliers) / len(gap_pts)
        inliers_mask = np.where(gap_mask)[0][gap_inliers]

        return fit_score, gap_score, inliers_mask

    def _end_points(self, pts_a):
        # TODO: faster
        dir = main_direction(pts_a[:,:2])
        d_pts = np.dot(pts_a[:,:2], dir)
        idx_a, idx_b = d_pts.argmin(), d_pts.argmax()
        return pts_a[idx_a], pts_a[idx_b]

    def _merge_clusters(self, points, clustering_labels_, max_angle_A, max_dist_A, max_angle_B, max_dist_B):

        cluster_labels, counts = np.unique(clustering_labels_, return_counts=True)
        cluster_labels_sorted = list(cluster_labels[1:][counts[1:].argsort()[::-1]])

        while len(cluster_labels_sorted) > 0:
            current_label = cluster_labels_sorted[0]
            candidate_labels = cluster_labels_sorted[1:]

            for candidate_label in candidate_labels:

                

                # Define current cluster
                maskCur = (clustering_labels_ == current_label)
                current_pts = points[maskCur]
                endpts_can = self._end_points(current_pts)
                current_v, current_i = pyrsc.Line().fit(current_pts, thresh=0.2, maxIteration=100)[:2]

                # Define candidate cluster
                maskCan = (clustering_labels_ == candidate_label)
                candidate_pts = points[maskCan]
                endpts_cur = self._end_points(candidate_pts)
                # TODO: pyrsc.Line().fit --> normal direction analysis? or faster
                candidate_v = pyrsc.Line().fit(candidate_pts, thresh=0.2, maxIteration=100)[0]

                # Define unlabeled
                mask_unlabelled = (clustering_labels_ == -1)

                # Order endpoints of clusters in line
                line_distances = cdist(endpts_cur, endpts_can)
                idx_cur, idx_can = np.unravel_index(line_distances.argmin(), line_distances.shape)
                p1_cur, p2_cur = endpts_cur[idx_cur], endpts_cur[1-idx_cur]
                p1_can, p2_can = endpts_can[idx_can], endpts_can[1-idx_can]
                merge_line = LineString([p1_cur,p1_can])
                merge_dist = np.min(line_distances)

                # Merge properties
                merge_angle = angle_between((p2_cur - p1_cur)[:2], (p1_can - p1_cur)[:2])
                dir_angle = angle_between(candidate_v[:2], current_v[:2])
                # print(candidate_label, 'to',current_label, ':', (round(merge_angle,2),(180 - max_angle_A)), round(dir_angle,2), (round(merge_dist,2),max_dist_A))
                
                if merge_angle > 90 and (dir_angle < max_angle_A or dir_angle > 180-max_angle_A) and merge_dist < max_dist_A:  
                    # 1. Clusters nearby
                    clustering_labels_[maskCan] = current_label
                    cluster_labels_sorted.remove(candidate_label)
                    fill_mask = poly_box_clip(points[mask_unlabelled], merge_line.buffer(.1), bottom=np.min([p1_cur[2],p1_can[2]]), top=np.max([p1_cur[2],p1_can[2]]))
                    clustering_labels_[mask_unlabelled][fill_mask] = current_label
                    # print('Added cluster', candidate_label, 'to',current_label)

                elif merge_angle > (180 - max_angle_B) and (dir_angle < max_angle_A or dir_angle > 180-max_angle_A) and merge_dist < max_dist_B: 
                    # 2. Cluster far away
                    fit_score, gap_score, gap_inliers = self._catenary_merge(points, maskCan, maskCur, mask_unlabelled)
                    if fit_score > .8:
                        clustering_labels_[maskCan] = current_label
                        cluster_labels_sorted.remove(candidate_label)
                        clustering_labels_[np.where(mask_unlabelled)[0][gap_inliers]] = current_label
                        # print('Added cluster', candidate_label, 'to',current_label)

            # Remove cluster from list
            cluster_labels_sorted.remove(current_label)
        
        return clustering_labels_


    def get_label_mask(self, points, labels, mask, tilecode):
        """
        Returns the label mask for cable classification.

        Parameters
        ----------
        points : array of shape (n_points, 3)
            The point cloud <x, y, z>.
        mask : array of shape (n_points,) with dtype=bool
            Pre-mask used as labelled cable points.
        tilecode : string

        Returns
        -------
        An array of shape (n_points,) with dtype=bool indicating which points
        belong to type tram cable class.
        """

        logger.info('Cable Extractor ' + f'(label={self.label})')

        label_mask = np.zeros((len(points),), dtype=bool)

        # 1. Candidata cable point selection
        linear_point_mask = self._get_linear_points(points[mask], self.neighborhood_method)

        # 2. Cluster linear points
        clustering_labels = self._get_cable_clusters(points[mask], 
                                                     linear_point_mask,
                                                     self.min_points)

        # 3. Grow cable clusters
        grown_clustering_labels = self._grow_clusters(points[mask], 
                                                      clustering_labels,
                                                      **self.grow_params)

        # 4. Merge cable clusters
        processed_clustering_labels = self._merge_clusters(points[mask], 
                                                           grown_clustering_labels, 
                                                           **self.merge_params)

        cable_mask = (processed_clustering_labels != -1)
        label_mask[mask] = cable_mask
        return label_mask

def catenary_func(x, a, b, c):
    return a + c * np.cosh((x-b) / c)