"""Search Space Reducer - Vertical Segmentation"""

import logging
import numpy as np

from src.abstract_processor import AbstractProcessor
from src.utils.interpolation import FastGridInterpolator
from src.utils.ahn_utils import fill_tile

logger = logging.getLogger(__name__)

class LowHeightFilter(AbstractProcessor):
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

    def __init__(self, label, ahn_reader, min_point_height=5):
        super().__init__(label)
        self.ahn_reader = ahn_reader
        self.min_point_height = min_point_height

    def get_label_mask(self, points, labels, mask, tilecode, cable_labels):
        """
        Returns the label mask for ground points.

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
        belong to type ground class.
        """

        logger.info('Search Space Reducer - Ground Filter' + f'(label={self.label})')

        label_mask = np.zeros((len(points),), dtype=bool)

        if mask is None:
            mask = np.ones((len(points),), dtype=bool)
        mask_ids = np.where(mask)[0]
        
        # Merge Ground and Artifact and Interpolate gaps of AHN tile
        ahn_tile = fill_tile(self.ahn_reader.filter_tile(tilecode))

        # Interpolate AHN ground data for points
        fast_z = FastGridInterpolator(ahn_tile['x'], ahn_tile['y'], ahn_tile['ground_surface'])
        ground_z = fast_z(points[mask])

        ground_mask = (points[mask, 2] < ground_z + self.min_point_height)
        label_mask[mask_ids[ground_mask]] = True

        return label_mask, cable_labels, None

class HighHeightFilter(AbstractProcessor):
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

    def __init__(self, label, ahn_reader, max_point_height=np.inf):
        super().__init__(label)
        self.ahn_reader = ahn_reader
        self.max_point_height = max_point_height

    def get_label_mask(self, points, labels, mask, tilecode, cable_labels):
        """
        Returns the label mask for sky points.

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
        belong to type building class.
        """

        logger.info('Search Space Reducer - Sky Filter' + f'(label={self.label})')

        label_mask = np.zeros((len(points),), dtype=bool)

        if mask is None:
            mask = np.ones((len(points),), dtype=bool)
        mask_ids = np.where(mask)[0]

        # Merge Ground and Artifact
        ahn_tile = self.ahn_reader.filter_tile(tilecode)
        if 'artifact_surface' in ahn_tile.keys():
            values = ahn_tile['ground_surface'].copy()
            values[np.isnan(values)] = ahn_tile['artifact_surface'][np.isnan(values)]
        else:
            values = ahn_tile['ground_surface'].copy()

        # Interpolate AHN ground data for points
        # TODO: time consuming? (interpolates ground for all points of tile)
        fast_z = FastGridInterpolator(ahn_tile['x'], ahn_tile['y'], values)
        ground_z = fast_z(points[mask])

        sky_mask = (points[mask, 2] > ground_z + self.max_point_height)
        label_mask[mask_ids[sky_mask]] = True

        return label_mask, cable_labels, None