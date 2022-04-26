"""Armatuur Fuser"""

import set_path
import numpy as np
import logging

from src.abstract_processor import AbstractProcessor
from src.region_growing.label_connected_comp import LabelConnectedComp
from milestone_2.cloud_utils import main_direction, angle_between
from src.utils.clip_utils import poly_clip

from scipy.stats import binned_statistic
from shapely.ops import nearest_points
from shapely.geometry import Point, Polygon, LineString

logger = logging.getLogger(__name__)

class ArmatuurFuser(AbstractProcessor):
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

    def __init__(self, label, armatuur_params={'width': (.25, 1.25), 'height': (.25, .6), 'z_offset':(0,.55), 'axis_offset': 0.1},
        min_cable_bending=2):
        super().__init__(label)
        self.armatuur_params = armatuur_params
        self.min_cable_bending = min_cable_bending

    def _is_armatuur(self, props, cable_angles):

        # 1. cluster dimensions
        props['width_dim_condition'] =\
              self.armatuur_params['width'][0] < props['width'] < self.armatuur_params['width'][1]
        props['height_dim_condition'] =\
            self.armatuur_params['height'][0] < props['height'] < self.armatuur_params['height'][1]

        # 2. centroid should be under cable
        props['z_offset'] =\
            self.armatuur_params['z_offset'][0] < props['z_offset'] < self.armatuur_params['z_offset'][1]
        props['axis_offset'] = props['axis_offset'] < self.armatuur_params['axis_offset']
        
        # 3. cable bending
        props['cable_bending'] = bool(cable_angles[np.abs(props['axis_pos'] - cable_angles[:,0]).argmin(), 1])

        return props

    def _cluster_properties(self, points, cable_axis, cable_zline, cable_axisline):
    
        # cluster props
        props = {}

        # Cluster dimension
        props['width'] = cable_axis.max() - cable_axis.min()
        props['height'] = points[:,2].max() - points[:,2].min()
        props['centroid'] = points.mean(axis=0)

        # Cluster position offset
        props['axis_pos'] = cable_axis.mean() # TODO: change to middle?
        props['z_offset'] = nearest_points(cable_zline, Point([props['axis_pos'], props['centroid'][2]]))[0].y - props['centroid'][2]
        props['axis_offset'] = LineString(nearest_points(cable_axisline, Point(props['centroid'][:2]))).length

        return props

    def _cable_angle(self, cable_zline, bin_dist=3):

        coords = np.asarray(cable_zline.coords)
        no_ = len(coords)
        angles = np.zeros(no_)
        if no_ > 2 * bin_dist:
            for i in range(bin_dist, no_ - bin_dist):
                v_a = coords[i] - coords[i - bin_dist]
                v_b = coords[i + bin_dist] - coords[i]
                angle = angle_between(v_a, v_b)
                if angle > self.min_cable_bending and v_a[1] < v_b[1]:
                    angles[i] = 1 #angle

        return np.vstack([coords[:,0],angles]).T

    def _clip_cable_area(self, points, cable_axis, cable_zline, cable_axisline, h_buffer=.5, w_buffer=.5):

        # Height clip
        cable_zpoly = cable_zline.buffer(h_buffer)
        dir_points = np.vstack((cable_axis, points[:,2])).T
        height_mask = poly_clip(dir_points, cable_zpoly)

        # Direction clip
        cable_axispoly = cable_axisline.buffer(w_buffer)
        axis_mask = poly_clip(points[:,:2], cable_axispoly)

        # Clip
        mask = height_mask & axis_mask

        return mask

    def _armatuur_extractor_lcc(self, points, cable_mask, cable_axis, cable_zline, cable_axisline):

        # 0. create mask
        armatuur_mask = np.zeros(len(points), dtype=bool)
        armatuur_labels = np.zeros(len(points))

        # 1. Clip search-area
        clip_mask = self._clip_cable_area(points, cable_axis, cable_zline, cable_axisline, 1, .5)

        # 2. LabelConnectedComponents
        mask = clip_mask & ~cable_mask
        lcc = LabelConnectedComp(label=-1, grid_size=0.07,
                                min_component_size=20)
        point_components = lcc.get_components(points[mask])
        

        # 3. Cluster analysis for armatuur
        cc_tags = {}
        cc_labels = set(np.unique(point_components)).difference((-1,0))
        if len(cc_labels) > 0:
            cable_angles = self._cable_angle(cable_zline)

            for cc in cc_labels:
                cc_mask = (point_components == cc)
                cc_props = self._cluster_properties(points[mask][cc_mask], cable_axis[mask][cc_mask], cable_zline, cable_axisline)
                cc_props = self._is_armatuur(cc_props, cable_angles)
                cc_tags[cc] = cc_props
                armatuur_labels[np.where(mask)[0][cc_mask]] = cc
                armatuur_mask[np.where(mask)[0][cc_mask]] = np.all(list(cc_props.values())[-5:-1])
   
        return armatuur_mask, armatuur_labels, cc_tags

    def _has_decoration(self, points, cable_mask, cable_axis, cable_zline, cable_axisline, threshold=.5, width_buffer=.2, plot=False):

        # create cable bottom polygon
        search_height = 1 # TODO: parameter analysis
        cable_radius = 0.1 # TODO: parameter analysis (we use different radius?)
        top_coords = np.array(cable_zline.coords) - [0, cable_radius]
        bottom_coords = np.array(cable_zline.coords) - [0, cable_radius + search_height]
        clip_poly = Polygon([*top_coords, *bottom_coords[::-1]])
        dir_points = np.vstack((cable_axis, points[:,2])).T
        
        # decoration clip
        # TODO: Pre clip with min Z and max Z? poly_clip is comp.exp.
        mask = np.all([
            ~cable_mask,
            poly_clip(points[:,:2], cable_axisline.buffer(width_buffer)), # optional to make have smaller width buffer
            poly_clip(dir_points, clip_poly), # clip points under the line
            cable_axis > .25, 
            cable_axis < cable_axis[cable_mask].max() - .25
            ], axis=0)

        if mask.sum() == 0:
            return False

        # density analysis
        res = binned_statistic(cable_axis[mask], np.ones(mask.sum()), statistic='count', bins=np.arange(0.25, cable_axis[cable_mask].max()-.25, 0.2)) # TODO: right margins
        pt_threshold = 10
        bin_density = res.statistic > pt_threshold
        cable_density_per = bin_density.sum() / len(bin_density)
        result = cable_density_per > threshold
            
        return result

    def _cable_cut(self, points, mask):
        """Create a new axis along the direction of a cable. Cable start is 0"""
        cable_dir = main_direction(points[mask][:,:2])
        cable_dir_axis = np.dot(points[:,:2], cable_dir)
        cable_dir_axis -= cable_dir_axis[mask].min()
        return cable_dir_axis

    def _linestring_cable_fit(self, points, cable_axis, binwidth_z=.75, binwidth_axis=.5):
        """
        Returns linetring fits for both z and xy projections.

        Parameters
        ----------
        points : array of shape (n_points, 3)
            The point cloud <x, y, z>.
        cable_axis : array of shape (n_points,)
            The cable directional axis values <d>.
        binwidth_z : float (default .75)
            The bindwithd used to calculate the statistic over.
        binwidth_axis : float (default .5)
            The bindwithd used to calculate the statistic over.

        Returns
        -------
        cable_zline : LineString
            linestring fit on Z axis projection.
        cable_axisline : LineString
            linestring fit on XY projection.
        """

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

    def _cable_fit(self, points, cable_mask, radius_h=.07, radius_w=.1, binwidth_z=.75, binwidth_axis=.5, refit=True):
        """
        Returns the label mask for the re-fitted cable.

        Parameters
        ----------
        points : array of shape (n_points, 3)
            The point cloud <x, y, z>.
        cable_mask : array of shape (n_points,) with dtype=bool
            Pre-mask used as labelled cable points.
        radius : float (default .07)
            Cable radius used for clipping points
        z_binwidth : float (default .75)
            The binwidth to average cable z for.
        axis_binwidth : float (default .5)
            The binwidth to average cable x,y for.

        Returns
        -------
        An array of shape (n_points,) with dtype=bool indicating which points
        should be labelled as cable points.
        """

        # Calculate directional axis
        cable_axis = self._cable_cut(points, cable_mask)
        dir_points = np.vstack([cable_axis,points[:,2]]).T
        cable_zline, cable_axisline = self._linestring_cable_fit(points[cable_mask], cable_axis[cable_mask], binwidth_z, binwidth_axis)

        # Estimate clip mask for cable points 
        if refit:
            h_mask = poly_clip(dir_points, cable_zline.buffer(radius_h)) # Analysis showed 20cm diameter
            axis_mask = poly_clip(points, cable_axisline.buffer(radius_w))
            cable_mask = h_mask & axis_mask
            cable_axis = self._cable_cut(points, cable_mask)
            cable_zline, cable_axisline = self._linestring_cable_fit(points[cable_mask], cable_axis[cable_mask], binwidth_z, binwidth_axis)

        return cable_mask, cable_axis, cable_zline, cable_axisline

    def get_label_mask(self, points, labels, mask, tilecode):
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

        # Set cable_mask
        cable_mask, cable_axis, cable_zline, cable_axisline = self._cable_fit(points, mask, refit=False)
    

        # Decoration check
        if not self._has_decoration(points, cable_mask, cable_axis, cable_zline, cable_axisline):
            armatuur_mask, armatuur_labels, cc_tags = self._armatuur_extractor_lcc(points, cable_mask, cable_axis, cable_zline, cable_axisline)
            label_mask[armatuur_mask] = True
        
        return label_mask