"""Visualisation utilities."""

import numpy as np
import pandas as pd
import folium
import os
import subprocess
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon, LineString

from ..utils import las_utils
from ..utils import bgt_utils
from ..labels import Labels
from ..utils.rd_converter import RDWGS84Converter

import open3d as o3d

bgt_labels = {'boom': 'Tree',
              'lichtmast': 'Lamp post',
              'verkeersbord': 'Traffic sign',
              'pand': 'Building',
              'wegdeel': 'Road',
              'bank': 'City bench',
              'afvalbak': 'Rubbish bin',
              'tram_rail': 'Tram Track'}
bgt_colors = {'boom': 'green',
              'lichtmast': 'orange',
              'verkeersbord': 'crimson',
              'pand': 'lightblue',
              'pand_poly': 'royalblue',
              'wegdeel': 'lightgrey',
              'bank': 'darkviolet',
              'afvalbak': 'pink',
              'tram_rail': 'yellow'}
cloud_colors = {'Unlabelled': 'lightgrey',
                'Ground': 'peru',
                'Building': 'lightblue',
                'Cable': 'green',
                'Light cable': 'green',
                'Tram cable': 'yellow',
                'Suspended streetlight': 'red',
                'Sky': 'whitesmoke',
                'Noise': 'whitesmoke'}


def plot_ahn_surface(ahn_tile, surf='ground_surface', ax=None, cbar_pad=0.04):
    cmap = {'ground_surface': 'gist_earth',
            'building_surface': 'Oranges'}
    cbar_title = {'ground_surface': 'Ground elevation (m)',
                  'building_surface': 'Building height (m)'}
    full_plot = False
    if ax is None:
        full_plot = True
        fig, ax = plt.subplots(1)

    x_min = int(ahn_tile['x'][0])
    x_max = int(np.ceil(ahn_tile['x'][-1]))
    y_min = int(ahn_tile['y'][-1])
    y_max = int(np.ceil(ahn_tile['y'][0]))

    im = ax.imshow(ahn_tile[surf], extent=[x_min, x_max, y_min, y_max],
                   interpolation='none', cmap=cmap[surf])

    ax.set_title(surf)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xticks(range(x_min, x_max+1, 10))
    ax.set_xticklabels(range(x_min, x_max+1, 10))
    ax.set_yticks(range(y_min, y_max+1, 10))
    ax.set_yticklabels(range(y_min, y_max+1, 10))
    cbar = plt.colorbar(im, ax=[ax], fraction=0.046, pad=cbar_pad)
    cbar.ax.set_ylabel(cbar_title[surf], rotation=270, labelpad=10)
    if full_plot:
        plt.show()

def plot_ahn_sidebyside(tilecode, ahn_reader):
    ahn_tile = ahn_reader.filter_tile(tilecode)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plot_ahn_surface(ahn_tile, surf='ground_surface', ax=ax1)
    plot_ahn_surface(ahn_tile, surf='building_surface', ax=ax2)
    ax2.set_yticklabels([])
    ax2.yaxis.label.set_visible(False)
    fig.suptitle(f'AHN tile {tilecode}', fontsize=14)
    plt.show()

def plot_ahn_merged(tilecode, ahn_reader):
    ahn_tile = ahn_reader.filter_tile(tilecode)
    fig, ax1 = plt.subplots(1, figsize=(7, 5))
    plot_ahn_surface(ahn_tile, surf='ground_surface', ax=ax1, cbar_pad=0.14)
    plot_ahn_surface(ahn_tile, surf='building_surface', ax=ax1, cbar_pad=0.04)
    ax1.set_title(f'AHN tile {tilecode}', fontsize=14)
    plt.show()

def plot_cloud_slice(las_file, ahn_reader, min_plane_height=4, max_plane_height=12,
                     hide_noise=False, ax=None, title=None, legend_below=False):
    full_plot = False
    if ax is None:
        fig, ax = plt.subplots(1, constrained_layout=True)
        full_plot = True

    las = las_utils.read_las(las_file)
    labels = las.label
    points = np.vstack((las.x, las.y, las.z)).T

    tilecode = las_utils.get_tilecode_from_filename(las_file)
    ((x_min, y_max), (x_max, y_min)) =\
        las_utils.get_bbox_from_tile_code(tilecode)

    points_z = ahn_reader.interpolate(tilecode, points,
                                      np.ones((len(points),), dtype=bool),
                                      'ground_surface')

    plane_mask = ((points[:, 2] >= points_z + min_plane_height)
                  & (points[:, 2] <= points_z + max_plane_height))
    label_set = np.unique(labels[plane_mask])

    for label in label_set:
        if label == Labels.NOISE and hide_noise:
            continue
        label_mask = plane_mask & (labels == label)
        label_str = Labels.get_str(label)
        ax.scatter(points[label_mask, 0], points[label_mask, 1],
                   c=cloud_colors[label_str], marker='.', label=label_str)

    box = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                            linewidth=1, linestyle='--', edgecolor='black',
                            fill=False)
    ax.add_patch(box)

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_xticks(range(x_min, x_max+1, 10))
    ax.set_xticklabels(range(x_min, x_max+1, 10))
    ax.set_ylabel('Y')
    ax.set_yticks(range(y_min, y_max+1, 10))
    ax.set_yticklabels(range(y_min, y_max+1, 10))
    if not legend_below:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.set_aspect('equal', adjustable='box')
    if full_plot:
        plt.show()

def plot_bag_bgt(tilecode, building_file=None, tram_file=None, ax=None, title=None, show_legend=True,
             legend_below=False, padding = 2.5, offset=0):
    
    full_plot = False
    if ax is None:
        width = 7 if show_legend else 5
        fig, ax = plt.subplots(1, figsize=(width, 5), constrained_layout=True)
        full_plot = True

    if title is None:
        title = f'Tile {tilecode}'

    tram_tracks = []
    buildings = []

    if building_file:
        buildings = bgt_utils.get_polygons(building_file, tilecode, padding=padding, offset=offset)
    if tram_file:
        tram_tracks = bgt_utils.get_linestrings(tram_file, tilecode, padding=padding)

    ((x_min, y_max), (x_max, y_min)) =\
        las_utils.get_bbox_from_tile_code(tilecode)

    for poly in [Polygon(bld) for bld in buildings]:
        x, y = poly.exterior.xy
        ax.fill(x, y, c='tab:red', alpha=.25, label='Building',
                zorder=-1)
        ax.plot(x, y, c='tab:red', zorder=0)

    for line in [LineString(track) for track in tram_tracks]:
        x, y = line.xy
        ax.plot(x, y, c='yellow', label='Tram track',
                zorder=1)

    box = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                            linewidth=1, linestyle='--', edgecolor='black',
                            fill=False)
    ax.add_patch(box)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if not legend_below:
            ax.legend(by_label.values(), by_label.keys(),
                      loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            ax.legend(by_label.values(), by_label.keys(),
                      loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    ax.set_xticks(range(x_min, x_max+1, 10))
    ax.set_xticklabels(range(x_min, x_max+1, 10))
    ax.set_yticks(range(y_min, y_max+1, 10))
    ax.set_yticklabels(range(y_min, y_max+1, 10))

    ax.set_xlim((x_min - padding, x_max + padding))
    ax.set_ylim((y_min - padding, y_max + padding))
    ax.set_aspect('equal', adjustable='box')
    if full_plot:
        plt.show()

def plot_bgt_bag_and_cloudslice(tilecode, las_file, ahn_reader,
                            building_file=None, tram_file=None,
                            min_plane_height=4, max_plane_height=12,
                            hide_noise=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.5))
    plot_bag_bgt(tilecode, building_file, tram_file, title='BGT data', ax=ax1,
             legend_below=True)
    plot_cloud_slice(las_file, ahn_reader, min_plane_height=min_plane_height,
                     max_plane_height=max_plane_height, hide_noise=hide_noise,
                     title='LAS labels', ax=ax2, legend_below=True)
    ax2.set_yticklabels([])
    ax2.yaxis.label.set_visible(False)
    fig.suptitle(f'Tile {tilecode}', fontsize=14)
    fig.subplots_adjust(top=1)
    plt.show()

def plot_tiles_map(tiles, train_tiles=[], width=1024, height=1024,
                   zoom_control=True, zoom_start=14, opacity=0.25):
    """
    Visualise the locations of all point cloud tiles in a given folder and
    overlay them on an OpenStreetMap of the area. The returned map is
    interactive, i.e. it allows panning and zooming, and tilecodes are
    displayed as tooltip on hoovering.
    """
    tile_df = (pd.DataFrame(columns=['Tilecode', 'X1', 'Y1', 'Train'])
               .set_index('Tilecode'))
    for tilecode in tiles:
        ((x1, _), (_, y1)) = las_utils.get_bbox_from_tile_code(tilecode)
        tile_df.loc[tilecode] = [x1, y1, False]
    for tilecode in train_tiles:
        tile_df.loc[tilecode, 'Train'] = True

    conv = RDWGS84Converter()

    center = conv.from_rd(int((tile_df.X1.max() + 50 + tile_df.X1.min()) / 2),
                          int((tile_df.Y1.max() + 50 + tile_df.Y1.min()) / 2))

    f = folium.Figure(width=width, height=height)

    # Create Folium background map.
    tiles_map = (folium.Map(location=center, tiles='cartodbpositron',
                            min_zoom=10, max_zoom=20, zoom_start=zoom_start,
                            zoom_control=zoom_control, control_scale=True)
                 .add_to(f))

    for index, row in tile_df.iterrows():
        rect = [conv.from_rd(row.X1, row.Y1),
                conv.from_rd(row.X1 + 50, row.Y1 + 50)]
        if row.Train:
            fc = 'darkorange'
            fop = opacity
        else:
            fc = 'royalblue'
            fop = 0.1
        (folium.Rectangle(bounds=rect, tooltip=index, color='royalblue',
                          weight=1, fill_color=fc, fill_opacity=fop)
         .add_to(tiles_map))

    return tiles_map

def save_tiles_map(map, outfile, width=1024, height=1024):
    """Save a given tiles map to a file, e.g. as PNG."""
    tmpfile = 'tmp.html'
    tmpurl = f'file://{os.getcwd()}/{tmpfile}'
    map.save(tmpfile)
    subprocess.check_call(['cutycapt', '--delay=1000',
                           f'--min-width={width}', f'--min-height={height}',
                           f'--url={tmpurl}', f'--out={outfile}'])
    if os.path.exists(tmpfile):
        os.remove(tmpfile)

def plot_pointcloud_o3d(points, labels, exclude_labels=(), pointcloud_rgb=None):

    open3d_pcd = o3d.geometry.PointCloud() # Initialize point cloud
    open3d_pcd.points = o3d.utility.Vector3dVector(points) # Give points

    colors = np.zeros((len(points),3))
    if pointcloud_rgb is not None:
        colors = pointcloud_rgb

    cc_labels = set(np.unique(labels)).difference(exclude_labels)

    if len(cc_labels) > 1:
        for label in set(np.unique(labels)).difference(exclude_labels):
            label_mask = labels == label
            label_color = np.random.random(3)
            colors[label_mask] = label_color
        
    open3d_pcd.colors = o3d.utility.Vector3dVector(colors) # Give colors

    o3d.visualization.draw_geometries([open3d_pcd]) # Visualize
