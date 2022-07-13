"""Cable Extractor"""

import time
import logging
import warnings
import numpy as np
import pandas as pd

import open3d as o3d
from loess import loess_1d
from pyntcloud import PyntCloud
from sklearn.cluster import DBSCAN
from shapely.geometry import LineString
from scipy.spatial import KDTree
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from scipy.spatial.distance import cdist
from scipy.optimize import OptimizeWarning

from ..abstract_processor import AbstractProcessor
from ..utils.clip_utils import poly_clip, poly_box_clip
from ..utils.cloud_utils import unit_vector
from ..analysis.analysis_tools import conf_matrix

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

    def __init__(self, label, voxel_size=.09, neigh_radius=.5, linearity_thres=.9,
                 max_v_angle=20, grow_radius=.3, max_merge_angle=3, 
                 min_segment_length=3, cable_height=.8, cable_width=.1,
                 cable_size=0.1, refit=False):
        super().__init__(label)
        self.voxel_size = voxel_size
        self.neigh_radius = neigh_radius
        self.linearity_thres = linearity_thres
        self.max_v_angle = max_v_angle
        self.grow_radius = grow_radius
        self.max_merge_angle = max_merge_angle
        self.min_segment_length = min_segment_length
        self.cable_height = cable_height
        self.cable_width = cable_width
        self.refit = refit
        self.cable_size=cable_size

    def _pointcloud_o3d(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def _voxelize(self, points, voxel_size):
        """ Returns the voxelization of a Point Cloud."""

        logger.info(f'Voxelizeing {len(points)} points')

        cloud = PyntCloud(pd.DataFrame(points, columns=['x','y','z']))
        voxelgrid_id = cloud.add_structure("voxelgrid", size_x=voxel_size, size_y=voxel_size, size_z=voxel_size, regular_bounding_box=False)
        voxel_grid = cloud.structures[voxelgrid_id]
        voxel_centers = voxel_grid.voxel_centers[np.unique(voxel_grid.voxel_n)]
        inv_voxel_idx = np.unique(voxel_grid.voxel_n, return_inverse=True)[1]

        return voxel_grid, voxel_centers, inv_voxel_idx

    def _principal_vector(self, points):
        cov = self._pointcloud_o3d(points).compute_mean_and_covariance()[1]
        eig_val, eig_vec = np.linalg.eig(cov)
        return eig_vec[:, eig_val.argmax()]

    # TODO: hybrid search?
    def _point_features(self, pcd, radius):
        
        #pcd.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50))
        pcd.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius))
        eig_val, eig_vec = np.linalg.eig(np.asarray(pcd.covariances))

        #: sort eigenvalues λ1 > λ2 > λ3
        index_array = np.flip(eig_val.argsort(), axis=1)
        eig_val = np.take_along_axis(eig_val, index_array, axis=1)

        linearity = np.nan_to_num((eig_val[:, 0] - eig_val[:, 1]) / eig_val[:, 0])

        principal_axis = eig_vec[np.arange(len(eig_vec)), :, index_array[:,0]]

        vertical_angle = np.degrees(np.arccos(np.clip(np.dot(principal_axis, [0.,0.,1.]), -1.0, 1.0)))
        vertical_angle = 90 - np.abs(vertical_angle - 90)

        return linearity, principal_axis, vertical_angle

    def _neighborhood_analysis(self, points, radius, linearity_thres, max_angle):

        pcd = self._pointcloud_o3d(points)
        linearity, principal_axis, vertical_angle = self._point_features(pcd, radius)
        candidate_mask = (linearity > linearity_thres) & (vertical_angle > (90-max_angle))

        return candidate_mask, principal_axis

    def _candidate_cable_points(self, points, voxel_size, radius, linearity_thres, max_angle):
        
        if voxel_size is not None:
            _, voxel_centers, inv_voxel_idx = self._voxelize(points, voxel_size)
            candidate_mask, principal_axis = self._neighborhood_analysis(voxel_centers, radius, linearity_thres, max_angle)
            
            # convert voxel features back to point features
            principal_axis = principal_axis[inv_voxel_idx]
            candidate_mask = candidate_mask[inv_voxel_idx]
        else:
            candidate_mask, principal_axis = self._neighborhood_analysis(points, radius, linearity_thres, max_angle)

        return candidate_mask, principal_axis

    def _grow_cluster_points(self, points, candidate_mask, labels, radius, principal_axis):

        cl_labels = np.full(len(points),-1)
        cl_labels[candidate_mask] = labels

        # Create KDTree
        kd_tree_candidate = KDTree(points[candidate_mask])
        kd_tree_other = KDTree(points[~candidate_mask])

        # Find neighbors of candidate points
        indices = kd_tree_candidate.query_ball_tree(kd_tree_other, r=radius)
        neighbors_n_parent = np.array([(j, i) for i in range(len(indices)) for j in indices[i]], dtype=int)
        if len(neighbors_n_parent) < 2:
            return cl_labels, np.zeros(len(points), dtype=bool), principal_axis
        neighbors_idx = np.unique(neighbors_n_parent[:,0], return_index=True)[0]
        neighbors_idx = np.where(~candidate_mask)[0][neighbors_idx]
        grow_mask = np.zeros(len(points), dtype=bool)
        grow_mask[neighbors_idx] = True

        # Assign candidate axis to non candidate neighbors
        nn_idx = kd_tree_candidate.query(points[grow_mask], distance_upper_bound=radius)[1]
        grow_parent_idx = np.where(candidate_mask)[0][nn_idx]
        principal_axis[grow_mask] = principal_axis[grow_parent_idx]

        # Assign candidate labels to non candidate neighbors
        cl_labels[grow_mask] = cl_labels[grow_parent_idx]

        return cl_labels, grow_mask, principal_axis

    def _get_end_points(self, points, principal_axis):
        d_pts = np.dot(points[:,:2], principal_axis[:2])
        idx_a, idx_b = d_pts.argmin(), d_pts.argmax()
        return np.array([points[idx_a], points[idx_b]])

    def _cable_cluster_feature(self, points):

        # Compute cluster directions
        principal_axis = self._principal_vector(points) # questionable since clutter of points can influence..
        direction = np.abs(np.abs(np.degrees(np.arctan2(principal_axis[1],principal_axis[0]))) - 90)
        
        end_points = self._get_end_points(points, principal_axis)
        length = np.linalg.norm(end_points[0] - end_points[1])

        result = {
            'counts': len(points),
            'principal_axis': principal_axis,
            'dir': direction,
            'end_points': end_points,
            'length': length
        }
        return result

    def _cable_cluster_features(self, points, labels, exclude_labels=[]):
        cable_clusters = {}

        for label in np.unique(labels):
            if label not in exclude_labels:
                cl_points = points[labels==label]
                cable_clusters[label] = self._cable_cluster_feature(cl_points)

        return cable_clusters

    def _project_cable_axis(self, points, cable_mask):
        principal_v = self._principal_vector(points[cable_mask])
        cable_axis = np.dot(points[:,:2], unit_vector(principal_v[:2]))
        cable_axis -= cable_axis[cable_mask].min()
        return cable_axis

    def _nearest_points(self, pts_a, pts_b):
        dist = cdist(pts_a, pts_b)
        idx_a, idx_b = np.unravel_index(np.argmin(dist), dist.shape)
        return idx_a, idx_b, dist.min()
        
    def _cluster_distance(self, cluster_a, cluster_b):
        a_end = cluster_a.get('end_points')
        b_end = cluster_b.get('end_points')
        return self._nearest_points(a_end, b_end)[2]

    def _cluster_merge_condition(self, points, ccl_dict, cl_labels, a, b, max_dist=4):
        a_end = ccl_dict[a].get('end_points')
        b_end = ccl_dict[b].get('end_points')
        idx_a, idx_b = self._nearest_points(a_end, b_end)[:2]
        O = a_end[idx_a]
        A = b_end[idx_b]

        # Option 1
        a_points_xy_axis = np.abs(np.dot((points[cl_labels==a,:2]-O[:2]), ccl_dict[a].get('principal_axis')[:2]))
        b = self._principal_vector(points[cl_labels==a][a_points_xy_axis < 1.5]) # must be unit vector

        # Option 2:
        # principal_axis of end point --> Not good for smaller radius...

        a = A - O
        a_1 = np.dot(a,b)*b
        a_2 = a-a_1 

        dir_dist = np.linalg.norm(a_1)
        offset_dist = np.linalg.norm(a_2)

        dist_bool = dir_dist < max_dist
        offset_bool = offset_dist < max(.2+ dir_dist * .1, .2)

        merge_bool = dist_bool and offset_bool

        return merge_bool, dir_dist, (O, A)

    def _catenary_merge(self, points, cl_labels, a, b, pt_a, pt_b, cable_width=.15):

        unmasked_idx = np.where(cl_labels<1)[0]
        merge_line = LineString([pt_a, pt_b])

        xy_clip_mask = poly_clip(points[(cl_labels==a)|(cl_labels==b),:2], merge_line.buffer(2, cap_style=3))
        fit_points = points[(cl_labels==a)|(cl_labels==b)][xy_clip_mask]
        principal_v = unit_vector((pt_b - pt_a)[:2])
        x_fit_points = np.dot(fit_points[:,:2], principal_v) 
        x_shift = np.min(x_fit_points)
        x_fit_points -= x_shift

        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            try:
                # Fit on cable
                popt, _ = curve_fit(catenary_func, x_fit_points, fit_points[:,2])

                # Evaluate fit on cable
                errors = abs(catenary_func(x_fit_points, *popt) - fit_points[:,2])
                fit_inliers = errors < self.cable_size
                fit_score = np.sum(fit_inliers) / len(fit_inliers)

                # Fit on gap
                xy_clip_mask = poly_clip(points[unmasked_idx,:2], merge_line.buffer(.1))
                gap_points = points[unmasked_idx[xy_clip_mask]]
                if len(gap_points) < 1:
                    return 0, 0, np.array([])
                x_gap_points = np.dot(gap_points[:,:2], principal_v) - x_shift

                # Evaluate fit on gap
                errors = abs(catenary_func(x_gap_points, *popt) - gap_points[:,2])
                gap_inliers = errors < cable_width/2
                gap_score = np.sum(gap_inliers) / len(gap_inliers)
                inlier_idx = unmasked_idx[xy_clip_mask][gap_inliers]

            except OptimizeWarning:
                # Do your other thing
                return 0, 0, None

        return fit_score, gap_score, inlier_idx

    def _box_merge(self, points, cl_labels, pt_a, pt_b, cable_width=.15):

        unmasked_idx = np.where(cl_labels<1)[0]

        merge_line = LineString([pt_a, pt_b]).buffer(cable_width)
        clip_mask = poly_box_clip(points[unmasked_idx], merge_line, bottom=np.min((pt_a[2], pt_b[2]))-cable_width, top=np.max((pt_a[2],pt_b[2]))+cable_width)

        merge_b = unit_vector(pt_b-pt_a)
        a = points[unmasked_idx[clip_mask]]-pt_a
        a_1 = np.dot(a, merge_b)[np.newaxis, :].T * merge_b
        a_2 = a-a_1
        dist = np.linalg.norm(a_2, axis=1)
        merge_idx = unmasked_idx[clip_mask][dist < cable_width/2]

        return merge_idx

    def _cable_merging(self, points, cl_labels, max_merge_angle=5):

        for cl in set(np.unique(cl_labels)).difference((-1,)):
            if np.sum(cl_labels==cl) < 4:
                cl_labels[cl_labels==cl] = -1

        ccl_dict = self._cable_cluster_features(points, cl_labels, [-1])
        cl_ordered = sorted(ccl_dict, key=lambda x: ccl_dict[x]["counts"], reverse=True)
        
        i = 0
        while len(cl_ordered) > i:
            main_cl = cl_ordered[i]
            candidate_cls = cl_ordered[i+1:]
            i += 1 

            search = True
            while search:
                search = False
                
                # Filter different direction
                direction = [ccl_dict[x].get('dir') for x in candidate_cls]
                angle_mask = np.abs(ccl_dict[main_cl].get('dir') - direction) < max_merge_angle
                selection_cls = np.asarray(candidate_cls)[angle_mask]

                if len(selection_cls) > 0:
                    # Filter range
                    ccls_dist = np.array([self._cluster_distance(ccl_dict[main_cl], ccl_dict[x]) for x in selection_cls])
                    ccls_ordered = ccls_dist.argsort()[:np.sum(ccls_dist<5)]
                    selection_cls = selection_cls[ccls_ordered]

                    for x in selection_cls:
                        valid_merge, dir_dist, (pt_a, pt_b) = self._cluster_merge_condition(points, ccl_dict, cl_labels, main_cl, x)
                        if valid_merge:
                            # Do your thing 
                            if dir_dist < 1:
                                inlier_mask = self._box_merge(points, cl_labels, pt_a, pt_b)
                            else:
                                fit_score, _, inlier_mask = self._catenary_merge(points, cl_labels, main_cl, x, pt_a, pt_b)
                                if fit_score < .8:
                                    continue
                            
                            # Assign new labels
                            cl_labels[inlier_mask] = main_cl
                            cl_labels[cl_labels==x] = main_cl

                            # Delete old cluster
                            ccl_dict.pop(x, None)
                            candidate_cls.remove(x)
                            cl_ordered.remove(x)

                            # Compute cluster features 
                            ccl_dict[main_cl] = self._cable_cluster_feature(points[cl_labels==main_cl])

                            search = True
                            break

        return cl_labels, ccl_dict

    def _cable_fit_loess(self, points, cable_axis):

        # z fit
        xnew = np.linspace(0, cable_axis.max(), int(cable_axis.max()/.2))
        _, zout, _ = loess_1d.loess_1d(cable_axis, points[:,2], xnew=xnew, degree=2, frac=0.25)
        line_pts = np.vstack((xnew, zout)).T
        cable_zline = LineString(line_pts)

        # xy fit
        xnew = np.linspace(0, cable_axis.max(), int(cable_axis.max()/.5))
        _, xout, _ = loess_1d.loess_1d(cable_axis, points[:,0], xnew=xnew, degree=2, frac=0.25)
        _, yout, _ = loess_1d.loess_1d(cable_axis, points[:,1], xnew=xnew, degree=2, frac=0.25)
        line_pts = np.vstack((xout, yout)).T
        cable_axisline = LineString(line_pts)

        return cable_zline, cable_axisline

    def _cable_fit(self, points, cable_axis, binwidth_z=.75, binwidth_axis=.5):

        cable_max = cable_axis.max()
        
        # LineString fit Z projection
        bins = np.linspace(0 - (binwidth_z/2), cable_max + (binwidth_z/2), int(round(cable_max/binwidth_z)+2))
        mean_z, bin_edges, _ = binned_statistic(cable_axis, points[:, 2], statistic='mean', bins=bins)
        x_coords = (bin_edges[:-1] + bin_edges[1:]) / 2
        line_pts = np.vstack((x_coords, mean_z)).T
        line_pts = line_pts[~np.isnan(line_pts).any(axis=1)]
        cable_zline = LineString(line_pts)

        # LineString fit XY projection
        bins = np.linspace(0 - (binwidth_axis/2), cable_max + (binwidth_axis/2), int(round(cable_max/binwidth_axis)+2))
        mean_x, _, _ = binned_statistic(cable_axis, points[:, 0], statistic='mean', bins=bins)
        mean_y, _, _ = binned_statistic(cable_axis, points[:, 1], statistic='mean', bins=bins)
        line_pts = np.vstack((mean_x, mean_y)).T
        line_pts = line_pts[~np.isnan(line_pts).any(axis=1)]
        cable_axisline = LineString(line_pts)

        return cable_zline, cable_axisline

    def _cable_refit(self, points, cable_mask, height_radius=.08, width_radius=.1, method='binned', refit=True):

        cable_axis = self._project_cable_axis(points, cable_mask)
        if method == 'loess':
            cable_zline, cable_axisline = self._cable_fit_loess(points[cable_mask], cable_axis[cable_mask])
        else:
            cable_zline, cable_axisline = self._cable_fit(points[cable_mask], cable_axis[cable_mask])

        # Estimate clip mask for cable points 
        if refit:
            cable_axis_points = np.vstack([cable_axis, points[:,2]]).T
            z_mask = poly_clip(cable_axis_points, cable_zline.buffer(self.cable_height)) # Analysis showed 10cm radius
            axis_mask = poly_clip(points, cable_axisline.buffer(self.cable_width)) # Analysis showed 8cm radius
            cable_mask = cable_mask | (z_mask & axis_mask) # old + new_points
            cable_axis = self._project_cable_axis(points, cable_mask)

            if method == 'loess':
                cable_zline, cable_axisline = self._cable_fit_loess(points[cable_mask], cable_axis[cable_mask])
            else:
                cable_zline, cable_axisline = self._cable_fit(points[cable_mask], cable_axis[cable_mask])

        return cable_mask, cable_axis, cable_zline, cable_axisline

    def get_label_mask(self, points, labels, mask, tilecode, cable_labels):
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
        cable_labels = np.zeros((len(points),), dtype=bool)
        cable_labels_full = np.full(len(points),-1, dtype=int)

        if mask is None:
            mask = np.ones((len(points),), dtype=bool)
        mask_ids = np.where(mask)[0]
        if len(mask_ids) < 20:
            return label_mask, cable_labels_full, None

        # 1. Canndidate Points
        logger.info(f'Candidate cable point selection')
        candidate_mask, principal_axis = self._candidate_cable_points(points[mask], voxel_size=self.voxel_size, radius=self.neigh_radius, linearity_thres=self.linearity_thres, max_angle=self.max_v_angle)
        if np.sum(candidate_mask) == 0:
            return label_mask, cable_labels_full, None

        # 2. Clustering
        logger.info('Clustering candidate cable points')
        clustering = (DBSCAN(eps=self.neigh_radius, min_samples=1, p=2).fit(points[mask][candidate_mask]))

        # 3. Grow cable clusters
        logger.info('Growing cables segments')
        cable_labels, _, principal_axis = self._grow_cluster_points(points[mask], candidate_mask, labels=clustering.labels_, radius=self.grow_radius, principal_axis=principal_axis)

        # 4. Merge cable clusters
        logger.info('Merging cables segments')
        cable_labels, ccl_dict = self._cable_merging(points[mask], cable_labels, max_merge_angle=self.max_merge_angle)
        short_clusters = [key for key in ccl_dict.keys() if ccl_dict[key]['length'] < self.min_segment_length]
        cable_labels[np.isin(cable_labels, short_clusters)] = -1

        # 5. Cable segment refit
        # logger.info('Cables refitting..')
        # for cl in set(np.unique(cable_labels)).difference((-1,)):
        #     cable_cl_mask = self._cable_refit(points[mask], cable_mask=(cable_labels==cl), refit=self.refit)[0]
        #     cable_labels[cable_cl_mask] = cl

        cable_labels_full[mask] = cable_labels
        cable_mask = (cable_labels > -1)
        label_mask[mask] = cable_mask

        return label_mask, cable_labels_full, None

    def get_analysis_mask(self, points, labels, mask, true=None, a=None):

        if true is None:
            true = np.zeros(len(points))

        label_mask = np.zeros((len(points),), dtype=bool)
        cable_labels_full = np.full(len(points),-1, dtype=int)

        mask = labels == 0

        confs = np.repeat(conf_matrix(true, labels)[np.newaxis,:,:], 5, axis=0)
        times = np.zeros(5)

        # 1. Canndidate Points
        start = time.time()
        candidate_mask, principal_axis = self._candidate_cable_points(points[mask], voxel_size=self.voxel_size, radius=self.neigh_radius, linearity_thres=self.linearity_thres, max_angle=self.max_v_angle)
        times[0] = time.time() - start
        if np.sum(candidate_mask) == 0:
            return label_mask, cable_labels_full, None, [times, confs]
        pred = labels.copy()

        pred[np.where(mask)[0][candidate_mask]] = self.label
        confs[0] = conf_matrix(true, pred)


        # 2. Clustering
        start = time.time()
        clustering = (DBSCAN(eps=self.neigh_radius, min_samples=1, p=2).fit(points[mask][candidate_mask]))
        times[1] = time.time() - start
        confs[2] = confs[1]

        # 3. Grow cable clusters
        start = time.time()
        cable_labels, grow_mask, principal_axis = self._grow_cluster_points(points[mask], candidate_mask, labels=clustering.labels_, radius=self.grow_radius, principal_axis=principal_axis)
        times[2] = time.time() - start
        pred = labels.copy()
        pred[np.where(mask)[0][cable_labels>-1]] = self.label
        confs[2] = conf_matrix(true, pred)


        # 4. Merge cable clusters
        start = time.time()
        cable_labels, ccl_dict = self._cable_merging(points[mask], cable_labels, max_merge_angle=self.max_merge_angle)
        
        # 4.2 Length condition
        short_clusters = [key for key in ccl_dict.keys() if ccl_dict[key]['length'] < self.min_segment_length]
        cable_labels[np.isin(cable_labels, short_clusters)] = -1
        times[3] = time.time() - start
        pred = labels.copy()
        pred[np.where(mask)[0][cable_labels>-1]] = self.label
        confs[3] = conf_matrix(true, pred)


        # 5. Cable segment refit
        # start = time.time()
        # for cl in set(np.unique(cable_labels)).difference((-1,)):
        #     cable_cl_mask = self._cable_refit(points[mask], cable_mask=(cable_labels==cl), refit=self.refit)[0]
        #     cable_labels[cable_cl_mask] = cl
        # times[4] = time.time() - start
        # pred = labels.copy()
        # pred[np.where(mask)[0][cable_labels>-1]] = self.label
        # confs[4] = conf_matrix(true, pred)

        cable_labels_full[mask] = cable_labels

        cable_mask = (cable_labels > -1)
        label_mask[mask] = cable_mask
        return label_mask, cable_labels_full, None, [times,confs]

def catenary_func(x, a, b, c):
    return a + c * np.cosh((x-b) / c)