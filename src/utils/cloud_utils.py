from pyntcloud import PyntCloud
import pandas as pd
import numpy as np
import open3d as o3d

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
    """ Returns the angle in degree between vectors 'v1' and 'v2'"""
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

def pointcloud_o3d(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def voxelize(points, voxel_size):
    """ Returns the voxelization of a Point Cloud."""

    cloud = PyntCloud(pd.DataFrame(points, columns=['x','y','z']))
    voxelgrid_id = cloud.add_structure("voxelgrid", size_x=voxel_size, size_y=voxel_size, size_z=voxel_size, regular_bounding_box=False)
    voxel_grid = cloud.structures[voxelgrid_id]
    voxel_centers = voxel_grid.voxel_centers[np.unique(voxel_grid.voxel_n)]
    inv_voxel_idx = np.unique(voxel_grid.voxel_n, return_inverse=True)[1]

    return voxel_grid, voxel_centers, inv_voxel_idx

def principal_vector(points):
    cov = pointcloud_o3d(points).compute_mean_and_covariance()[1]
    eig_val, eig_vec = np.linalg.eig(cov)
    return eig_vec[:, eig_val.argmax()]