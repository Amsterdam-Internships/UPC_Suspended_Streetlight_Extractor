"""Search Space Reducer"""

from cv2 import FILE_STORAGE_FORMAT_MASK
import set_path
import numpy as np
import logging
from shapely.geometry import Polygon, Point

from src.abstract_processor import AbstractProcessor
from src.utils.las_utils import get_polygon_from_tile_code, get_bbox_from_tile_code
from src.utils.clip_utils import poly_clip
from src.labels import Labels

import open3d as o3d
from scipy.spatial import distance
from src.utils.interpolation import FastGridInterpolator

logger = logging.getLogger(__name__)


class SearchSpaceReducer(AbstractProcessor):
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

    def __init__(self, label, ahn_tile, bgt_reader, min_ground_height=4.5,
                 min_building_height=4.5, building_offset=1.5, padding=0,
                 remove_outliers=False):
        super().__init__(label)
        self.ahn_tile = ahn_tile
        self.bgt_reader = bgt_reader
        self.building_offset = building_offset
        self.min_ground_height = min_ground_height
        self.min_building_height = min_building_height
        self.padding = padding
        self.remove_outliers = remove_outliers

    def _ground_filter(self, points, mask, min_height):
        """ Based on certain properties clip ground points.  """

        ground_mask = np.zeros(len(points), dtype=bool)

        # Merge Ground and Artifact
        if 'artifact_surface' in self.ahn_tile.keys():
            values = self.ahn_tile['ground_surface'].copy()
            values[np.isnan(values)] = self.ahn_tile['artifact_surface'][np.isnan(values)]
        else:
            values = self.ahn_tile['ground_surface'].copy()

        # Interpolate AHN ground data for points
        # TODO: time consuming? (interpolates ground for all points of tile)
        fast_z = FastGridInterpolator(self.ahn_tile['x'], self.ahn_tile['y'], values)
        ground_z = fast_z(points[mask])

        filter_mask = (points[mask, 2] < ground_z + min_height)
        ground_mask[np.where(mask)[0]] = filter_mask
        
        return ground_mask

    def _sampled_ground_elevation(self, gridsize=10):
        ground_subsample = self.ahn_tile['ground_surface'][::gridsize,::gridsize]
        mask_ids = np.isfinite(ground_subsample)
        ground_z = ground_subsample[mask_ids]
        ground_coords = np.array(np.meshgrid(self.ahn_tile['x'][::gridsize], self.ahn_tile['y'][::gridsize]))[:, mask_ids]
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

    def _building_filter(self, points, mask, tilecode,
                         offset, padding, min_height):
        """
        tilecode : str
            Tilecode to use for this filter.
        bgt_reader : BGTPolyReader object
            Used to load building polygons.
        ahn_reader : AHNReader object
            AHN data will be used to set a minimum height for each building polygon.
        offset : int (default: 0)
            The footprint polygon will be extended by this amount (in meters).
        padding : float (default: 0)
            Optional padding (in m) around the tile when searching for objects.
        min_height : float (default: 4.5)
            AHN min building elevation cut-off.
        """

        # 1. Create mask and get ids of non-labelled
        building_mask = np.zeros(len(points), dtype=bool)
        mask_ids = np.where(mask)[0]

        # READ BGT
        building_polygons = self.bgt_reader.filter_tile(
                                    tilecode, bgt_types=['pand'],
                                    padding=padding, offset=offset,
                                    merge=True)
        if len(building_polygons) == 0: 
            return building_mask

        ground_points = self._sampled_ground_elevation()
        ahn_bld_z = FastGridInterpolator(self.ahn_tile['x'], self.ahn_tile['y'], self.ahn_tile['building_surface']) 
        # ahn_bld_z = ahn_reader.interpolate(tilecode, )
        tile_bbox = get_bbox_from_tile_code(tilecode)
        tile_polygon = get_polygon_from_tile_code(tilecode, 0)

        filter_mask = np.zeros(len(mask_ids), dtype=bool)
        for polygon in building_polygons:
            if polygon.intersects(tile_polygon):

                # 2 Calculate building height.
                center = np.asarray(polygon.centroid.coords)
                bld_ground_level = self._closest_point_cdist(center, ground_points)
                bld_points = self._random_polygon_points(polygon, size=20, offset=offset, bbox=tile_bbox)
                bld_height = np.nanmax(ahn_bld_z(bld_points))
                bld_height = bld_height - bld_ground_level[2]

                if bld_height > min_height:
                    # TODO if there are multiple buildings we could mask the points
                    # iteratively to ignore points already labelled.
                    clip_mask = poly_clip(points[mask][~filter_mask, :], polygon)
                    filter_mask[np.where(~filter_mask)[0][clip_mask]] = True

        # 3. AHN building roof elevation
        building_mask[mask_ids] = filter_mask

        return building_mask

    def _outlier_filter(self, points, mask, method='radius', voxelize=False):
        """
        The work of this region growing algorithm is based on the comparison
        of the angles between the points normals.
        The same can also be performed in Python using scipy.spatial.cKDTree
        with query_ball_tree or query.
        """

        # 1. Create mask and get ids of non-labelled
        noise_mask = np.zeros(len(points),dtype=bool)
        mask_ids = np.where(mask)[0]

        # Convert point cloud
        coords = np.vstack((points[mask, 0], points[mask, 1], points[mask, 2])).transpose()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)

        # Voxelize point cloud
        if voxelize: 
            pcd, _, trace_ids = pcd.voxel_down_sample_and_trace(voxel_size=.1, min_bound=pcd.get_min_bound(), max_bound=pcd.get_max_bound())
            trace_ids = np.asarray(trace_ids)

        # Remove outliers
        # TODO: Parameter Tuning
        if method == 'voxel_radius':
            _, non_outlier_ids = pcd.remove_radius_outlier(nb_points=1, radius=0.5)

        elif method == 'sor':
            _, non_outlier_ids = pcd.remove_statistical_outlier(nb_neighbors=4, std_ratio=1)

        outlier_mask = np.ones(len(coords), dtype=bool)
        if voxelize:
            non_outlier_ids = trace_ids[non_outlier_ids].flatten()
        outlier_mask[non_outlier_ids] = False
        noise_mask[mask_ids] = outlier_mask

        return noise_mask

    def get_label_mask(self, points, labels, mask, tilecode):

        logger.info('Search Space Reducer' + f'(label={self.label})')

        label_mask = np.ones((len(points),), dtype=bool)
        labels = np.zeros(len(points))

        # Step 1: Height Filter
        ground_mask = self._ground_filter(points, label_mask, self.min_ground_height)
        labels[ground_mask] = Labels.GROUND
        label_mask[ground_mask] = False

        # Step 2: BGT filter
        building_mask = self._building_filter(points, label_mask, tilecode, self.building_offset, self.padding, self.min_building_height)
        labels[building_mask] = Labels.BUILDING
        label_mask[building_mask] = False

        # Step 3: Outlier filter
        if self.remove_outliers:
            noise_mask = self._outlier_filter(points, label_mask, method='sor')
            labels[noise_mask] = Labels.NOISE
            label_mask[noise_mask] = False

        return label_mask, labels