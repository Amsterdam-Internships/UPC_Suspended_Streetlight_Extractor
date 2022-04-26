"""TramTrack Fuser"""

import set_path
import numpy as np
import logging
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon

from src.abstract_processor import AbstractProcessor
from src.utils.math_utils import compute_bounding_box
from src.labels import Labels

logger = logging.getLogger(__name__)


class TramTrackFuser(AbstractProcessor):
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

    def _label_tramcable_like_clusters(self, points, ground_z, clustering,
                                       tramtrack_polygon, min_height):
        """ Based on certain properties of a cable we label clusters.  """

        cable_mask = np.zeros(len(points), dtype=bool)
        cable_count = 0

        cc_labels = np.unique(clustering)

        cc_labels = set(cc_labels)

        for cl in cc_labels:
            # select points that belong to the cluster
            c_mask = (clustering == cl)

            target_z = ground_z[c_mask]
            cc_pts = points[c_mask]

            # TODO: use linestring over polygon
            cbbox = compute_bounding_box(cc_pts[:,:2])
            cable_poly = Polygon([cbbox[:2], (cbbox[0], cbbox[3]), cbbox[2:], (cbbox[2], cbbox[1])])
            
            # Rule based classification
            if cable_poly.intersects(tramtrack_polygon):
                cc_height = cc_pts[:, 2] - target_z
                if cc_height.min() < min_height:
                    cable_mask = cable_mask | c_mask
                    cable_count += 1

        logger.debug(f'{cable_count} cables labelled.')

        return cable_mask


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
            The tile code.

        Returns
        -------
        An array of shape (n_points,) with dtype=bool indicating which points
        belong to type tram cable class.
        """

        # logger.info('Tramtrack fuser ' +
        #             f'(label={self.label}, {Labels.get_str(self.label)}).')
        logger.info('Tramtrack fuser ' + f'(label={self.label})')

        label_mask = np.zeros((len(points),), dtype=bool)

        tramtrack_polygons = self.bgt_reader.filter_tile(
                    tilecode, padding=10)

        if len(tramtrack_polygons) == 0:
            logger.debug('No tram track parts found for tile, skipping.')
            return label_mask

        # AHN ground interpolation
        ground_z = self.ahn_reader.interpolate(
                            tilecode, points[mask], mask, 'ground_surface')

        # TODO: ParameterTune
        clustering = (DBSCAN(eps=1, min_samples=4, p=2).fit(points[mask]))

        tramtrack_polygon = self._tramtrack_polygon(tramtrack_polygons, self.track_buffer)
        tramcable_mask = self._label_tramcable_like_clusters(points[mask], ground_z,
                                                   clustering.labels_, 
                                                   tramtrack_polygon,
                                                   self.min_height)

        label_mask[mask] = tramcable_mask

        return label_mask