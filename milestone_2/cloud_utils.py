import set_path

from pyntcloud import PyntCloud
import pandas as pd
import numpy as np
import open3d as o3d

from multiprocessing import Pool

def main_direction(points):
    """ Returns the eigenvector corresponding to the largest eigenvalue of `points`"""
    cov = np.cov(points, rowvar=False)
    eig_val, eig_vec = np.linalg.eig(cov)
    dir_v = eig_vec[:,eig_val.argmax()]
    if dir_v[0] < 0:
        dir_v *= -1
    return dir_v

def unit_vector(v1):
    """ Returns the unit vector of `v1`"""
    return v1 / np.linalg.norm(v1)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def cable_cut(points, mask=None):
    """Create a new axis along the direction of a cable. Cable start is 0"""
    if mask is None:
        mask = np.ones((len(points),), dtype=bool)

    cable_dir = main_direction(points[mask][:,:2])
    cable_dir_axis = np.dot(points[:,:2], cable_dir)
    cable_dir_axis -= cable_dir_axis[mask].min()
    return cable_dir_axis

def decompose(cov):
    return np.linalg.eig(cov) if cov.shape == (3,3) else (None, None)

def voxelize(points, voxel_size, return_voxel_points=False, logger=False):
    """ Returns the voxelization of a Point Cloud."""

    # Voxelize point cloud
    cloud = PyntCloud(pd.DataFrame(points, columns=['x','y','z']))
    voxelgrid_id = cloud.add_structure("voxelgrid", size_x=voxel_size, size_y=voxel_size, size_z=voxel_size, regular_bounding_box=False)
    voxel_grid = cloud.structures[voxelgrid_id]
    if logger:
        print('Voxels per axis:',voxel_grid.x_y_z)
        print('Voxel size:',voxel_grid.shape)

    # Group points per voxel 
    if return_voxel_points:
        pv_table = np.vstack([voxel_grid.voxel_n, np.arange(0,voxel_grid.voxel_n.shape[0])]).T
        pv_table = pv_table[pv_table[:, 0].argsort()]
        voxel_idx, voxel_idx_index = np.unique(pv_table[:, 0], return_index=True)
        voxel_points = np.split(pv_table[:, 1], voxel_idx_index[1:])

        return voxel_grid, voxel_points, voxel_idx
    
    return voxel_grid

def neighborhood_pca(points, method='radius', knn=30, radius=1, voxel_size=1):
    """
    Returns the eigen values and vectors of points in a Point Cloud.
    
    Parameters
    ----------
    `points` : array of shape `(n_points, 3)`
        The point cloud <x, y, z>.
    `method` : str
        The method to use to define a points neighborhood.
    `knn` : int (default= 30)
        k Nearest Neighbors parameter.
    `radius` : int (default= 1)
        Radius parameter.
    `voxel_size` : int (default= 1)
        voxelization parameter.

    Returns
    -------
    An array of shape (n_points, 2) indicating eigen values and vectors
    """

    #TODO: pyntcloud --> eigenvalues sneller?

    if method == 'knn':
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
        covariances = np.asarray(pcd.covariances)
        with Pool() as pool:
            eig_val_vec = pool.map(decompose, covariances)

    elif method == 'radius':
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius))
        covariances = np.asarray(pcd.covariances)
        with Pool() as pool:
            eig_val_vec = pool.map(decompose, covariances)

    elif method == 'voxel':
        voxel_grid, voxel_point_idx, voxel_idx = voxelize(points, voxel_size=voxel_size, return_voxel_points=True)
        voxel_table = {k: v for v, k in enumerate(voxel_idx)}
        covariances = [np.cov(points[idx], rowvar=False) for idx in voxel_point_idx]
        with Pool() as pool:
            voxel_eig_val_vec = pool.map(decompose, covariances)
        eig_val_vec = [voxel_eig_val_vec[voxel_table[voxel_id]] for voxel_id in voxel_grid.voxel_n]
    
    return eig_val_vec