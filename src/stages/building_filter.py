"""Search Space Reducer - Building Filter"""

import logging
import numpy as np

from ..utils.interpolation import FastGridInterpolator
from ..abstract_processor import AbstractProcessor
from ..utils.las_utils import get_polygon_from_tile_code, get_bbox_from_tile_code
from ..utils.clip_utils import poly_clip
from ..labels import Labels

from shapely.geometry import Point
from scipy.spatial import distance

logger = logging.getLogger(__name__)


class BuildingFilter(AbstractProcessor):
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

    def __init__(self, label, ahn_reader, bgt_reader, building_offset=1.5):
        super().__init__(label)
        self.ahn_reader = ahn_reader
        self.bgt_reader = bgt_reader
        self.building_offset = building_offset

    def _sampled_ground_elevation(self, ahn_tile, gridsize=10):
        ground_subsample = ahn_tile['ground_surface'][::gridsize,::gridsize]
        mask_ids = np.isfinite(ground_subsample)
        ground_z = ground_subsample[mask_ids]
        ground_coords = np.array(np.meshgrid(ahn_tile['x'][::gridsize], ahn_tile['y'][::gridsize]))[:, mask_ids]
        ground_points = np.vstack((ground_coords,ground_z)).T
        return ground_points

    def _closest_point_cdist(self, point, ground_points):
        closest_index = distance.cdist(point, ground_points[:,:2]).argmin()
        return ground_points[closest_index]

    def _random_polygon_points(self, polygon, size=10, offset=0, bbox=None):
        min_x, min_y, max_x, max_y = polygon.buffer(-offset).bounds
        roof_coords = []

        # limit polygon bounds to bbox bounds
        if bbox is not None:
            ((box_x_min, box_y_max), (box_x_max, box_y_min)) = bbox
            min_x = max(min_x,box_x_min)
            min_y = max(min_y,box_y_min)
            max_x = min(max_x,box_x_max)
            max_y = min(max_y,box_y_max)

        while len(roof_coords) < size:
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            if polygon.contains(Point(x,y)):
                roof_coords.append([x,y])

        return np.asarray(roof_coords)

    def _building_filter(self, points, tilecode,
                         offset):
        """
        tilecode : str
            Tilecode to use for this filter.
        bgt_reader : BGTPolyReader object
            Used to load building polygons.
        ahn_reader : AHNReader object
            AHN data will be used to set a minimum height for each building polygon.
        offset : int (default: 0)
            The footprint polygon will be extended by this amount (in meters).
        """

        # 1. Create mask and get ids of non-labelled
        building_mask = np.zeros(len(points), dtype=bool)

        # READ BGT
        building_polygons = self.bgt_reader.filter_tile(
                                    tilecode, bgt_types=['pand'],
                                    padding=offset, offset=offset,
                                    merge=True)

        logger.info(f'Buildings in tile {len(building_polygons)}')
        if len(building_polygons) == 0: 
            return building_mask

        # TODO: use cache
        ahn_tile = self.ahn_reader.filter_tile(tilecode)
        tile_polygon = get_polygon_from_tile_code(tilecode, 0)

        for polygon in building_polygons:
            if polygon.intersects(tile_polygon):
                # TODO if there are multiple buildings we could mask the points
                # iteratively to ignore points already labelled.
                clip_mask = poly_clip(points, polygon)
                building_mask = building_mask | clip_mask

        return building_mask

    def get_label_mask(self, points, labels, mask, tilecode, cable_labels):
        """
        Returns the label mask for buildings.

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

        logger.info('Search Space Reducer - Building Filter' + f'(label={self.label})')

        label_mask = np.zeros((len(points),), dtype=bool)

        if mask is None:
            mask = np.ones((len(points),), dtype=bool)
        mask_ids = np.where(mask)[0]

        building_mask = self._building_filter(points[mask], tilecode, self.building_offset)
        label_mask[mask_ids[building_mask]] = True

        return label_mask, cable_labels, None