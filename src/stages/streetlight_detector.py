"""Armatuur Fuser"""

import logging
import numpy as np

from scipy import ndimage
from scipy.stats import binned_statistic
from shapely.geometry import LineString
from sklearn.cluster import DBSCAN

from ..abstract_processor import AbstractProcessor
from ..utils.cloud_utils import main_direction, angle_between, voxelize
from ..utils.math_utils import minimum_bounding_rectangle, compute_bounding_box
from ..utils.clip_utils import poly_clip
from ..analysis.analysis_tools import get_objectwise_stats
from ..labels import Labels

logger = logging.getLogger(__name__)

class StreetlightDetector(AbstractProcessor):
    """
    Parameters
    ----------
    label : int
        Class label to use for this fuser.
    """

    def __init__(self, label, armatuur_params={'width': (.20, 1), 'height': (.15, 1.),
        'axis_offset': 0.15}, min_cable_bending=2, voxel_size=0.05, cable_sag_span=2):
        super().__init__(label)
        self.armatuur_params = armatuur_params
        self.min_cable_bending = min_cable_bending
        self.voxel_size = voxel_size
        self.cable_sag_span = cable_sag_span

    def _clip_cable_area(self, points, cable_yline, cable_zline, h_buffer=.5, w_buffer=.5):

        cable_zpoly = cable_zline.buffer(h_buffer, cap_style=3)
        height_mask = poly_clip(points[:,[0,2]], cable_zpoly)

        # Direction clip
        cable_axispoly = cable_yline.buffer(w_buffer, cap_style=3)
        axis_mask = poly_clip(points[:,[0,1]], cable_axispoly)

        # Clip
        mask = height_mask & axis_mask

        return mask

    def _pc_cable_rotation(self, points, mask):
        direction = main_direction(points[mask][:,:2])
        cable_dir_axis = np.dot(points[:,:2], direction)

        # rotation matrix
        theta = np.arctan2(direction[1],direction[0])
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s, 0), (s, c, 0),(0,0,1)))

        points_rotated = points.copy()
        points_rotated[:,:2] -= cable_dir_axis[mask].min() * direction
        points_rotated = points_rotated.dot(R)

        return points_rotated

    def _fit_linestring(self, points, bin_width=.75):
        """
        Returns linetring fits for both z and xy projections.

        Parameters
        ----------
        points : array of shape (n_points, 2)
            The point cloud <x, y, z>.
        binwidth : float (default .75)
            The bindwithd used to calculate the statistic over.

        Returns
        -------
        cable_line : LineString
            linestring fit on Y axis.
        """

        line_max = points[:,0].max()
        
        # LineString fit Z projection
        bins = np.linspace(0 - (bin_width/2), line_max + (bin_width/2),
                             int(round(line_max/bin_width)+2))
        means, bin_edges, _ = binned_statistic(points[:,0], points[:, 1],
                             statistic='mean', bins=bins)
        x_coords = (bin_edges[:-1] + bin_edges[1:]) / 2
        line_pts = np.vstack((x_coords, means)).T
        line_pts = line_pts[~np.isnan(line_pts).any(axis=1)]
        cable_line = LineString(line_pts)
        
        return cable_line

    def _compute_saggign_angle(self, x, z, span, d, fill=np.inf):
        d = int(span/d)
        bendings = np.full(len(x), fill)
        for i in range(len(x)):
            if i - d >=0 and i + d < len(x):
                O = np.array([0, z[i+1]])
                v_a = np.array([-span, z[i-d]]) - O
                v_b = np.array([span, z[i+d]]) - O
                bendings[i] = 180 - angle_between(v_a, v_b)
        return bendings

    def _search_armaturen(self, points, cable_mask):

        # parameters
        slice_width = 3
        armatuurs = []

        points_rotated = self._pc_cable_rotation(points, cable_mask)
        cable_yline = self._fit_linestring(points_rotated[cable_mask][:,[0,1]])
        cable_zline = self._fit_linestring(points_rotated[cable_mask][:,[0,2]])
        
        # 1. Clip Cable Area
        clip_mask = self._clip_cable_area(points_rotated, cable_yline, cable_zline, 1, 1)
        search_mask = clip_mask & ~cable_mask

        if np.sum(search_mask) < 10:
            return armatuurs

        # 3. Voxelize
        voxel_grid = voxelize(points_rotated[search_mask], self.voxel_size)[0]
        voxel_space = voxel_grid.get_feature_vector()

        # 4. Gridify Cable LineStrings
        min_x = voxel_grid.voxel_centers[0][0]
        max_x = voxel_grid.voxel_centers[-1][0]
        x_ = np.arange(min_x,max_x+2*voxel_grid.sizes[0],voxel_grid.sizes[0])
        z_ = np.interp(x_,cable_zline.xy[0],cable_zline.xy[1])
        y_ = np.interp(x_,cable_yline.xy[0],cable_yline.xy[1])
        a_ = self._compute_saggign_angle(x_, z_, self.cable_sag_span, voxel_grid.sizes[0])

        # 5. Loop through slices
        attachment_voxel_space = np.zeros(voxel_space.shape).flatten()
        for i in range(0, voxel_space.shape[0], slice_width):

            # 5.1 Slice Density Analysis
            row_slice = voxel_space[i:i+slice_width].sum(axis=0)>0

            t = int((z_[i+1] - voxel_grid.voxel_centers[0][2]) / voxel_grid.sizes[0]) + 1
            if np.sum(row_slice[:,:t]) > 5: # Check for points below cable

                # 5.2 Morophology filter on cable slice
                row_slice_closed = np.pad(row_slice, 2)
                row_slice_closed = ndimage.binary_dilation(row_slice_closed, iterations=2)
                row_slice_closed = ndimage.binary_erosion(row_slice_closed, iterations=2)
                row_slice_closed = row_slice_closed[2:-2,2:-2]

                # 5.3 Label Connected Components
                lcc, n_lcc = ndimage.label(row_slice_closed)
                for l in range(1,n_lcc+1):

                    cl = np.vstack(np.where(lcc==l)).T
                    if len(cl) > 5:

                        # 5.4 Component Boundingbox Analysis
                        (x_min, y_min, x_max, y_max) = compute_bounding_box(cl)
                        y_center = int(np.round(y_min + (y_max-y_min)/2))
                        x_center = int(np.round(x_min + (x_max-x_min)/2))
                        box_width = (x_max-x_min)*voxel_grid.sizes[0]
                        box_heigth = (y_max-y_min)*voxel_grid.sizes[0]
                        cl_center = voxel_grid.voxel_centers[np.ravel_multi_index((min(voxel_space.shape[0]-1,i+1),
                                                                x_center,y_center),voxel_space.shape)]
                        target_center = np.array([y_[i+1],z_[i+1]-(box_heigth/2)])
                        z_off = z_[i+1]-cl_center[2]
                        axis_off = np.abs(target_center[0]-cl_center[1])

                        # cond = [box_width >= self.armatuur_params['width'][0],
                        #     box_width < self.armatuur_params['width'][1],
                        #     box_heigth >= self.armatuur_params['height'][0],
                        #     box_heigth < self.armatuur_params['height'][1],
                        #     axis_off < self.armatuur_params['axis_offset'],
                        #     z_off > max(.1, box_heigth/2),
                        #     a_[i+1] > self.min_cable_bending]
                        # logger.info(f'{[box_width, box_heigth, axis_off]}')
                        # logger.info(f'{i}, {cond}')

                        if box_width >= self.armatuur_params['width'][0] and \
                            box_width < self.armatuur_params['width'][1] and \
                            box_heigth >= self.armatuur_params['height'][0] and \
                            box_heigth < self.armatuur_params['height'][1]  and \
                            axis_off < self.armatuur_params['axis_offset'] and \
                            z_off > max(.1, box_heigth/2) and \
                            a_[i+1] > self.min_cable_bending:

                            cl_indices = np.repeat((lcc==l)[np.newaxis,:,:], 3, axis=0)
                            
                            cl_indices = np.pad(cl_indices, ((min(i,1),1),(0,0),(0,0)))
                            cl_indices = ndimage.binary_dilation(cl_indices, iterations=1)
                            index_start = np.ravel_multi_index((max(i-1,0),0,0), voxel_space.shape)
                            cl_indices = index_start + np.where(cl_indices.flatten())[0]

                            # add attachment to space
                            attachment_voxel_space[cl_indices[cl_indices < len(attachment_voxel_space)]] = 1

        # 6. Label Connected Components [Attachment Grid]
        arm_lcc, arm_n_lcc = ndimage.label(attachment_voxel_space.reshape(voxel_space.shape))
        for arm_l in range(1,arm_n_lcc+1):
            arm_idx = np.isin(voxel_grid.voxel_n, np.where((arm_lcc==arm_l).flatten())[0])
            arm_mask = np.zeros(len(search_mask),dtype=bool)
            arm_mask[np.where(search_mask)[0][arm_idx]] = True

            # Bounding box analysis
            mbr, _, min_dim, max_dim, center = minimum_bounding_rectangle(points[arm_mask,:2])
            if min_dim > self.armatuur_params['width'][0] and \
                max_dim < self.armatuur_params['width'][1]:
                
                min_z, max_z = np.min(points[arm_mask,2]), np.max(points[arm_mask,2])
                z_dim = (max_z - min_z)
                z_center = min_z + z_dim/2
                center = np.hstack((center,z_center))
                armatuur = {
                    'center':center,
                    'min_bound_rec': mbr,
                    'mask': arm_mask,
                    'dims': (min_dim,max_dim,z_dim)
                }
                armatuurs.append(armatuur)

        return armatuurs

    def get_label_mask(self, points, labels, mask, tilecode, cable_labels):
        """
        Returns the label mask for armatuur classification.

        Parameters
        ----------
        points : array of shape (n_points, 3)
            The point cloud <x, y, z>.
        mask : array of shape (n_points,) with dtype=bool
            Pre-mask used as labelled cable points.
        tilecode : string
            The tile code.

        Returns
        -------
        An array of shape (n_points,) with dtype=bool indicating which points
        belong to type armatuur class.
        """

        logger.info('Armatuur fuser ' + f'(label={self.label})')

        label_mask = np.zeros((len(points),), dtype=bool)
        label_objects = []

        # Set mask & labels
        mask = (labels == Labels.CABLE)

        # Cable segmentation
        if cable_labels is None:
            cable_labels = np.full(len(points),-1, dtype=int)

            # Error no cables
            clustering = (DBSCAN(eps=.5, min_samples=1, p=2).fit(points[mask]))
            cable_labels[mask] = clustering.labels_


        cable_count = 0
        for cl in set(np.unique(cable_labels[mask])).difference((-1,)):
            cable_count += 1

            # select points that belong to the cluster
            cable_mask = (cable_labels == cl)

            if np.sum(cable_mask) > 100:
                cable_object = {'type': 'cable',
                                 'data': {'type': 'Utility', 'data': cable_mask}
                                } # TODO: add z-values to linestring
                extracted_armaturen = self._search_armaturen(points, cable_mask)

                if len(extracted_armaturen) > 0:
                    logger.info(f'Streetlight found!')
                    cable_object['data']['type'] = 'streetlight'
                    for armatuur in extracted_armaturen:
                        label_mask[armatuur['mask']] = True
                        label_objects.append({'type': 'armatuur', 'data': armatuur['mask']})

                label_objects.append(cable_object)
        
        logger.info(f'Analyzed number of cables: {cable_count}')
        
        return label_mask, cable_labels, label_objects

    def get_analysis_mask(self, points, labels, cable_labels, true_labels=None):

        label_mask = np.zeros((len(points),), dtype=bool)

        # Set mask & labels
        mask = (labels == Labels.CABLE)

        if true_labels is None:
            true_labels = np.zeros(len(points))

        cable_count = 0
        for cl in set(np.unique(cable_labels[mask])).difference((-1,)):
            cable_count += 1

            # select points that belong to the cluster
            cable_mask = (cable_labels == cl)

            if np.sum(cable_mask) < 100:
                continue

            try:
                extracted_armaturen = self._search_armaturen(points, cable_mask)
            
                # Decoration check
                for armatuur in extracted_armaturen:
                    label_mask[armatuur['mask']] = True
            except:
                continue

        true_mask = true_labels > 14

        report = {}
        report['TP'] = np.sum(true_mask & label_mask)
        report['FP'] = np.sum(~true_mask & label_mask)
        report['FN'] = np.sum(true_mask & ~label_mask)

        # Objectwise
        pred_labels = np.zeros(len(points))
        pred_labels[label_mask] = 15
        obj_stats = get_objectwise_stats(points, pred_labels, true_labels)
        
        report['TP_obj'] = obj_stats[2][0]
        report['FP_obj'] = obj_stats[2][1]
        report['FN_obj'] = obj_stats[2][2]

        return label_mask, report