"""Data Fusion Pipeline"""

from cProfile import label
# from tkinter import Label
# from unittest import result
import numpy as np
import os
import pathlib
import time
import logging
from tqdm import tqdm
import pandas as pd

from src.utils import las_utils
from src.analysis import analysis_tools
from src.labels import Labels

logger = logging.getLogger('src')


class Pipeline:
    """
    Pipeline for data fusion. The class accepts a list of DataFuser objects and
    processes a single point cloud or a folder of pointclouds by applying the
    given DataFusers consecutively. It is assumed that the fusers are ordered
    by importance: points labelled by each fuser are excluded from further
    processing.

    Parameters
    ----------
    process_sequence : iterable of type AbstractProcessor
        The processors to apply, in order.
    exclude_labels : list
        List of labels to exclude from processing.
    ahn_reader : AHNReader object
        Pointer to the AHNReader object used in the Processors. Required if
        caching is used.
    caching : bool (default: True)
        Enable caching of AHN interpolation data.
    """

    FILE_TYPES = ('.LAS', '.las', '.LAZ', '.laz')

    def __init__(self, processors=[], exclude_labels=[],
                 ahn_reader=None, caching=True):
        if ahn_reader is None and caching:
            logger.error(
                'An ahn_reader must be specified when caching is enabled.')
            raise ValueError
        self.processors = processors
        self.exclude_labels = exclude_labels
        self.ahn_reader = ahn_reader
        self.caching = caching
        if self.caching:
            self.ahn_reader.set_caching(self.caching)

    def _create_mask(self, mask, labels):
        """Create mask based on `exclude_labels`."""
        if mask is None:
            mask = np.ones((len(labels),), dtype=bool)
        if len(self.exclude_labels) > 0:
            for exclude_label in self.exclude_labels:
                mask = mask & (labels != exclude_label)
        return mask

    def process_cloud(self, tilecode, points, labels, mask=None):
        """
        Process a single point cloud.

        Parameters
        ----------
        tilecode : str
            The CycloMedia tile-code for the given pointcloud.
        points : array of shape (n_points, 3)
            The point cloud <x, y, z>.
        labels : array of shape (n_points, 1)
            All labels as int values
        mask : array of shape (n_points,) with dtype=bool
            Pre-mask used to label only a subset of the points.

        Returns
        -------
        An array of shape (n_points,) with dtype=uint16 indicating the label
        for each point.
        """
        mask = self._create_mask(mask, labels)
        if self.caching:
            self.ahn_reader.cache_interpolator(
                                tilecode, points, surface='ground_surface')

        cable_labels = None
        label_object_lists = []

        for obj in self.processors:
            start = time.time()
            label_mask, cable_labels, label_objects = obj.get_label_mask(points, labels, mask, tilecode, cable_labels)[:3]
            labels[label_mask] = obj.get_label()
            mask[label_mask] = False
            if label_objects is not None:
                label_object_lists.extend(label_objects)
            duration = time.time() - start
            logger.info(f'Processor finished in {duration:.2f}s, ' +
                        f'{np.count_nonzero(label_mask)} points labelled.')

        # return cable_labels, labels
        return labels, label_object_lists

    def process_file(self, in_file, in_labels=True, out_file=None, mask=None):
        """
        Process a single LAS file and save the result as .laz file.

        Parameters
        ----------
        in_file : str
            The file to process.
        out_file : str (default: None)
            The name of the output file. If None, the input will be
            overwritten.
        mask : array of shape (n_points,) with dtype=bool
            Pre-mask used to label only a subset of the points.
        """
        logger.info(f'Processing file {in_file}.')
        start = time.time()
        if not os.path.isfile(in_file):
            logger.error('The input file specified does not exist')
            return None

        if out_file is None:
            out_file = in_file

        tilecode = las_utils.get_tilecode_from_filename(in_file)
        pointcloud = las_utils.read_las(in_file)
        points = np.vstack((pointcloud.x, pointcloud.y, pointcloud.z)).T

        if in_labels and 'label' in pointcloud.point_format.extra_dimension_names:
            labels = pointcloud.label
        else:
            labels = np.zeros((len(points),), dtype='uint16')
            
        labels, _ = self.process_cloud(tilecode, points, labels, mask)
        las_utils.label_and_save_las(pointcloud, labels, out_file)

        duration = time.time() - start
        stats = analysis_tools.get_label_stats(labels)
        logger.info('STATISTICS\n' + stats)
        logger.info(f'File processed in {duration:.2f}s, ' +
                    f'output written to {out_file}.\n' + '='*20)

    def process_folder(self, in_folder, out_folder=None, in_prefix='',
                       out_prefix='', suffix='', hide_progress=False, in_labels=True):
        """
        Process a folder of LAS files and save each processed file.

        Parameters
        ----------
        in_folder : str or Path
           The input folder.
        out_folder : str or Path (default: None)
           The name of the output folder. If None, the output will be written
           to the input folder.
        in_prefix : str
            Optional prefix to filter files in the input folder. Only files
            starting with this prefix will be processed.
        out_prefix : str
            Optional prefix to prepend to output files. If an in_prefix is
            given, it will be replaced by the out_prefix.
        suffix : str or None
            Suffix to add to the filename of processed files. A value of None
            indicates that the same filename is kept; when out_folder=None this
            means each file will be overwritten.
        """
        if not os.path.isdir(in_folder):
            logger.error('The input path specified does not exist')
            return None
        if type(in_folder) == str:
            in_folder = pathlib.Path(in_folder)
        if out_folder is None:
            out_folder = in_folder
        else:
            pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)
        if suffix is None:
            suffix = ''

        logger.info('===== PIPELINE =====' +
                    f'Processing folder {in_folder}, ' +
                    f'writing results in {out_folder}.')

        files = [f for f in in_folder.glob('*')
                 if f.name.endswith(self.FILE_TYPES)
                 and f.name.startswith(in_prefix)]
        files_tqdm = tqdm(files, unit="file",
                          disable=hide_progress, smoothing=0)
        logger.debug(f'{len(files)} files found.')

        for file in files_tqdm:
            files_tqdm.set_postfix_str(file.name)
            filename, extension = os.path.splitext(file.name)
            if in_prefix and out_prefix:
                filename = filename.replace(in_prefix, out_prefix)
            elif out_prefix:
                filename = out_prefix + filename
            outfile = os.path.join(out_folder, filename + suffix + extension)
            self.process_file(file.as_posix(), in_labels, outfile)

        logger.info(f'Pipeline finished, {len(files)} processed.\n' + '='*20)

    def analyse_file(self, in_file, gt_file, in_labels=True, mask=None):
        """
        Process a single LAS file and save the result as .laz file.

        Parameters
        ----------
        in_file : str
            The file to process.
        out_file : str (default: None)
            The name of the output file. If None, the input will be
            overwritten.
        mask : array of shape (n_points,) with dtype=bool
            Pre-mask used to label only a subset of the points.
        """

        logger.info(f'Analyzing file {in_file}.')

        if not os.path.isfile(gt_file):
            logger.error('The ground true file specified does not exist')
            return None
        true_labels = las_utils.read_las(gt_file).label

        if not os.path.isfile(in_file):
            logger.error('The input file specified does not exist')
            return None

        tilecode = las_utils.get_tilecode_from_filename(in_file)
        pointcloud = las_utils.read_las(in_file)
        points = np.vstack((pointcloud.x, pointcloud.y, pointcloud.z)).T

        if in_labels and 'label' in pointcloud.point_format.extra_dimension_names:
            labels = pointcloud.label
        else:
            labels = np.zeros((len(points),), dtype='uint16')
            
        start = time.time()
        labels, label_object_lists = self.process_cloud(tilecode, points, labels, mask)
        duration = time.time() - start

        logger.info(f'list objects length:{len(label_object_lists)}')

        objects = {'cable':[],'armatuur':[]}
        for o in label_object_lists:
            objects[o['type']].append(o['data'])

        # TODO: Can make as param input
        report = analysis_tools.get_cable_stats_m3(labels, true_labels)
        report['tilecode'] = tilecode
        report['time'] = duration

        return report
        
    def analyse_folder(self, in_folder, gt_prefix='labelled_', in_prefix='labelled_', in_suffix='', in_labels=True, hide_progress=False):
        """
        Process a folder of LAS files and save each processed file.

        Parameters
        ----------
        in_folder : str or Path
           The input folder.
        out_folder : str or Path (default: None)
           The name of the output folder. If None, the output will be written
           to the input folder.
        in_prefix : str
            Optional prefix to filter files in the input folder. Only files
            starting with this prefix will be processed.
        out_prefix : str
            Optional prefix to prepend to output files. If an in_prefix is
            given, it will be replaced by the out_prefix.
        suffix : str or None
            Suffix to add to the filename of processed files. A value of None
            indicates that the same filename is kept; when out_folder=None this
            means each file will be overwritten.
        """
        if type(in_folder) == str:
            in_folder = pathlib.Path(in_folder)

        logger.info('===== PIPELINE =====' +
                    f'Processing folder {in_folder}, ')

        files = [f for f in in_folder.glob('*')
                 if f.name.endswith(tuple([in_suffix + t for t in self.FILE_TYPES]))
                 and f.name.startswith(in_prefix)]
        files_tqdm = tqdm(files, unit="file",
                          disable=hide_progress, smoothing=0)
        logger.debug(f'{len(files)} files found.')

        results = []
        # objects = {'cable':[],'armatuur':[]}

        for file in files_tqdm:
            files_tqdm.set_postfix_str(file.name)
            filename, extension = os.path.splitext(file.name)
            tilecode = las_utils.get_tilecode_from_filename(filename)

            gt_file = os.path.join(in_folder, gt_prefix + tilecode + extension)
            file_results = self.analyse_file(file.as_posix(), gt_file, in_labels)

            # objects['cable'].extend(file_objects['cable'])
            # objects['armatuur'].extend(file_objects['armatuur'])

            results.append(file_results)

        # pd.DataFrame(objects['cable']).to_csv('cable_objects.csv')
        # pd.DataFrame(objects['armatuur']).to_csv('armatuur_objects.csv')

        logger.info(f'Pipeline finished, {len(files)} analized.\n' + '='*20)

        return results





