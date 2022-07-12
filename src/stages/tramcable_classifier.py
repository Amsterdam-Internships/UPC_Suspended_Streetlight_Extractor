"""Tram CableClassifier"""

import logging
import numpy as np

from scipy.stats import binned_statistic
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon, LineString

from ..abstract_processor import AbstractProcessor
from ..utils.cloud_utils import main_direction
from ..labels import Labels

logger = logging.getLogger(__name__)


class TramCableClassifier(AbstractProcessor):
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

    def __init__(self, label, ahn_reader, bgt_reader,
                 track_buffer=2, min_height=7.5):
        super().__init__(label)
        self.ahn_reader = ahn_reader
        self.bgt_reader = bgt_reader
        self.track_buffer = track_buffer
        self.min_height = min_height

    def _tramtrack_polygon(self, tram_tracks, track_buffer):
        """ Get Polygon of the tram tracks in tile."""

        # Create tramtrack polygon
        tramtracks_poly = Polygon([])
        for track in tram_tracks:
            tramtracks_poly = tramtracks_poly.union(track)

        # Grow and erode tramtrack polygon
        tramtracks_poly = tramtracks_poly.buffer(2.8+track_buffer).buffer(-2.8)

        return tramtracks_poly

    def _cable_cut(self, points):
        """Create a new axis along the direction of a cable. Cable start is 0"""
        cable_dir = main_direction(points[:,:2])
        cable_dir_axis = np.dot(points[:,:2], cable_dir)
        cable_dir_axis -= cable_dir_axis.min()
        return cable_dir_axis

    def _linestring_cable_fit(self, points, cable_axis, binwidth_axis=.5):
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

        # LineString fit XY projection
        bins = np.linspace(0 - (binwidth_axis/2), cable_max + (binwidth_axis/2), int(round(cable_max/binwidth_axis)+2))
        mean_x, _, _ = binned_statistic(cable_axis, points[:, 0], statistic='mean', bins=bins)
        mean_y, _, _ = binned_statistic(cable_axis, points[:, 1], statistic='mean', bins=bins)
        line_pts = np.vstack((mean_x, mean_y)).T
        line_pts = line_pts[~np.isnan(line_pts).any(axis=1)]
        cable_axisline = LineString(line_pts)

        return cable_axisline

    def _label_tramcable_like_clusters(self, points, ground_z, cable_labels,
                                       tramtrack_polygon, min_height):
        """ Based on certain properties of a cable we label clusters.  """


        cable_mask = np.zeros(len(points), dtype=bool)
        cable_objects = []
        cable_count = 0

        for cl in set(np.unique(cable_labels)).difference((-1,)):
            # select points that belong to the cluster
            c_mask = (cable_labels == cl)

            target_z = ground_z[c_mask]
            cc_pts = points[c_mask]

            # TODO: use linestring over polygon
            cable_axis = self._cable_cut(cc_pts)
            cable_axisline = self._linestring_cable_fit(cc_pts, cable_axis)
            
            # Rule based classification
            if tramtrack_polygon.intersects(cable_axisline.buffer(.5)):
                cc_height = cc_pts[:, 2] - target_z
                if cc_height.min() < min_height:
                    cable_mask = cable_mask | c_mask
                    cable_objects.append({'type': 'cable', 'data': {'type': 'tram', 'linestring': cable_axisline.xy}}) # TODO: add z-values to linestring
                    cable_count += 1

        logger.debug(f'{cable_count} cables labelled.')

        return cable_mask, cable_objects

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
            The tile code.

        Returns
        -------
        An array of shape (n_points,) with dtype=bool indicating which points
        belong to type tram cable class.
        """

        logger.info('Tram cable classifier ' + f'(label={self.label})')

        label_mask = np.zeros((len(points),), dtype=bool)
        tram_cable_objects = []

        tramtrack_polygons = self.bgt_reader.filter_tile(
                    tilecode, padding=10)

        logger.debug(f'{len(tramtrack_polygons)} tram tracks in tile.')
        if len(tramtrack_polygons) == 0:
            return label_mask, cable_labels, tram_cable_objects
        
        # Set mask
        mask = (labels == Labels.CABLE)

        # AHN ground interpolation
        ground_z = self.ahn_reader.interpolate(
                            tilecode, points[mask], mask, 'ground_surface')

        # Cable segmentation
        if cable_labels is None:
            cable_labels = np.full(len(points),-1, dtype=int)

            clustering = (DBSCAN(eps=1, min_samples=1, p=2).fit(points[mask]))
            cable_labels[mask] = clustering.labels_

        tramtrack_polygon = self._tramtrack_polygon(tramtrack_polygons, self.track_buffer)
        tramcable_mask, tram_cable_objects = self._label_tramcable_like_clusters(points[mask], ground_z,
                                                   cable_labels[mask], 
                                                   tramtrack_polygon,
                                                   self.min_height)

        label_mask[mask] = tramcable_mask

        return label_mask, cable_labels, tram_cable_objects