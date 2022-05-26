# Copyright (c) OpenMMLab. All rights reserved.
import copy
import tempfile
import warnings
from os import path as osp
from typing import Callable, List, Optional, Union

import mmcv
import numpy as np
from mmengine.dataset import BaseDataset

from mmdet3d.datasets import DATASETS
from ..core.bbox import get_box_type
from .pipelines import Compose
from .utils import extract_result_dict, get_loading_pipeline


@DATASETS.register_module()
class Det3DDataset(BaseDataset):
    """Base Class of 3D dataset.

    This is the base dataset of SUNRGB-D, ScanNet, nuScenes, and KITTI
    dataset.
    # TODO: doc link here for the standard data format

    Args:
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to None.
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_prefix (dict, optional): Prefix for training data. Defaults to
            dict(pts='velodyne', img="").
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input, it usually has following keys.

                - use_camera: bool
                - use_lidar: bool
            Defaults to `dict(use_lidar=True, use_camera=False)`
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR'. Available options includes

            - 'LiDAR': Box in LiDAR coordinates, usually for
              outdoor point cloud 3d detection.
            - 'Depth': Box in depth coordinates, usually for
              indoor point cloud 3d detection.
            - 'Camera': Box in camera coordinates, usually
              for vision-based 3d detection.

        filter_empty_gt (bool, optional): Whether to filter the data with
            empty GT. Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """

    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(pts='velodyne', img=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 box_type_3d: dict = 'LiDAR',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 file_client_args: dict = dict(backend='disk'),
                 **kwargs):
        # init file client
        self.file_client = mmcv.FileClient(**file_client_args)
        self.filter_empty_gt = filter_empty_gt
        _default_modality_keys = ('use_lidar', 'use_camera')
        if modality is None:
            modality = dict()

        # Defaults to False if not specify
        for key in _default_modality_keys:
            if key not in modality:
                modality[key] = False
        self.modality = modality
        assert self.modality['use_lidar'] or self.modality['use_camera'], (
            'Please specify the `modality` (`use_lidar` '
            f' or `use_camera`) for {self.__class__.__name__}')

        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)

        if metainfo is not None and 'CLASSES' in metainfo:
            # we allow to train on subset of self.METAINFO['CLASSES']
            # map unselected labels to -1
            self.label_mapping = {
                i: -1
                for i in range(len(self.METAINFO['CLASSES']))
            }
            self.label_mapping[-1] = -1
            for label_idx, name in enumerate(metainfo['CLASSES']):
                ori_label = self.METAINFO['CLASSES'].index(name)
                self.label_mapping[ori_label] = label_idx
        else:
            self.label_mapping = {
                i: i
                for i in range(len(self.METAINFO['CLASSES']))
            }
            self.label_mapping[-1] = -1

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs)

    def _remove_dontcare(self, ann_info):
        """Remove annotations that do not need to be cared.

        -1 indicate dontcare in MMDet3d.

        Args:
            ann_info (dict): Dict of annotation infos. The
                instance with label `-1` will be removed.

        Returns:
            dict: Annotations after filtering.
        """
        img_filtered_annotations = {}
        filter_mask = ann_info['gt_labels_3d'] > -1
        for key in ann_info.keys():
            img_filtered_annotations[key] = (ann_info[key][filter_mask])
        return img_filtered_annotations

    def get_ann_info(self, index: int) -> dict:
        """Get annotation info according to the given index.

        Use index to get the corresponding annotations, thus the
        evalhook could use this api.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information.
        """
        data_info = self.get_data_info(index)
        # test model
        if 'ann_info' not in data_info:
            ann_info = self.parse_ann_info(data_info)
        else:
            ann_info = data_info['ann_info']

        return ann_info

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`

        In `Custom3DDataset`, we simply concatenate all the field
        in `instances` to `np.ndarray`, you can do the specific
        process in subclass. You have to convert `gt_bboxes_3d`
        to different coordinates according to the task.

        Args:
            info (dict): Info dict.

        Returns:
            dict: Processed `ann_info`
        """
        # add s or gt prefix for most keys after concat
        name_mapping = {
            'bbox_label': 'gt_labels',
            'bbox_label_3d': 'gt_labels_3d',
            'bbox': 'gt_bboxes',
            'bbox_3d': 'gt_bboxes_3d',
            'depth': 'depths',
            'center_2d': 'centers_2d',
            'attr_label': 'attr_labels'
        }

        instances = info['instances']
        keys = list(instances[0].keys())
        ann_info = dict()
        for ann_name in keys:
            temp_anns = [item[ann_name] for item in instances]
            if 'label' in ann_name:
                temp_anns = [self.label_mapping[item] for item in temp_anns]
            temp_anns = np.array(temp_anns)
            if ann_name in name_mapping:
                ann_name = name_mapping[ann_name]
            ann_info[ann_name] = temp_anns
        return ann_info

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Convert all relative path of needed modality data file to
        the absolute path. And process
        the `instances` field to `ann_info` in training stage.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """

        if self.modality['use_lidar']:
            info['lidar_points']['lidar_path'] = \
                osp.join(
                    self.data_prefix.get('pts', ''),
                    info['lidar_points']['lidar_path'])

        if self.modality['use_camera']:
            for cam_id, img_info in info['images'].items():
                if 'img_path' in img_info:
                    img_info['img_path'] = osp.join(
                        self.data_prefix.get('img', ''), img_info['img_path'])

        if not self.test_mode:
            info['ann_info'] = self.parse_ann_info(info)

        return info

    def prepare_data(self, index):
        """Data preparation for both training and testing stage.

        Called by `__getitem__`  of dataset.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)

        # deepcopy here to avoid inplace modification in pipeline.
        input_dict = copy.deepcopy(input_dict)

        # box_type_3d (str): 3D box type.
        input_dict['box_type_3d'] = self.box_type_3d
        # box_mode_3d (str): 3D box mode.
        input_dict['box_mode_3d'] = self.box_mode_3d

        # pre-pipline return None to random another in `__getitem__`
        if not self.test_mode and self.filter_empty_gt:
            if len(input_dict['ann_info']['gt_labels_3d']) == 0:
                return None

        example = self.pipeline(input_dict)
        if not self.test_mode and self.filter_empty_gt:
            # after pipeline drop the example with empty annotations
            # return None to random another in `__getitem__`
            if example is None or len(example['gt_labels_3d']) == 0:
                return None
        return example

    def format_results(self,
                       outputs,
                       pklfile_prefix=None,
                       submission_prefix=None):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (outputs, tmp_dir), outputs is the detection results,
                tmp_dir is the temporal directory created for saving json
                files when ``jsonfile_prefix`` is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
            out = f'{pklfile_prefix}.pkl'
        mmcv.dump(outputs, out)
        return outputs, tmp_dir

    def evaluate(self,
                 results,
                 metric=None,
                 iou_thr=(0.25, 0.5),
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluate.

        Evaluation in indoor protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str], optional): Metrics to be evaluated.
                Defaults to None.
            iou_thr (list[float]): AP IoU thresholds. Defaults to (0.25, 0.5).
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict: Evaluation results.
        """
        from mmdet3d.core.evaluation import indoor_eval
        assert isinstance(
            results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos)
        assert isinstance(
            results[0], dict
        ), f'Expect elements in results to be dict, got {type(results[0])}.'
        gt_annos = [info['annos'] for info in self.data_infos]
        label2cat = {i: cat_id for i, cat_id in enumerate(self.CLASSES)}
        ret_dict = indoor_eval(
            gt_annos,
            results,
            iou_thr,
            label2cat,
            logger=logger,
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d)
        if show:
            self.show(results, out_dir, pipeline=pipeline)

        return ret_dict

    # TODO check this where does this method is used
    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        raise NotImplementedError('_build_default_pipeline is not implemented '
                                  f'for dataset {self.__class__.__name__}')

    # TODO check this where does this method is used
    def _get_pipeline(self, pipeline):
        """Get data loading pipeline in self.show/evaluate function.

        Args:
            pipeline (list[dict]): Input pipeline. If None is given,
                get from self.pipeline.
        """
        if pipeline is None:
            if not hasattr(self, 'pipeline') or self.pipeline is None:
                warnings.warn(
                    'Use default pipeline for data loading, this may cause '
                    'errors when data is on ceph')
                return self._build_default_pipeline()
            loading_pipeline = get_loading_pipeline(self.pipeline.transforms)
            return Compose(loading_pipeline)
        return Compose(pipeline)

    # TODO check this where does this method is used
    def _extract_data(self, index, pipeline, key, load_annos=False):
        """Load data using input pipeline and extract data according to key.

        Args:
            index (int): Index for accessing the target data.
            pipeline (:obj:`Compose`): Composed data loading pipeline.
            key (str | list[str]): One single or a list of data key.
            load_annos (bool): Whether to load data annotations.
                If True, need to set self.test_mode as False before loading.

        Returns:
            np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor]:
                A single or a list of loaded data.
        """
        assert pipeline is not None, 'data loading pipeline is not provided'
        # when we want to load ground-truth via pipeline (e.g. bbox, seg mask)
        # we need to set self.test_mode as False so that we have 'annos'
        if load_annos:
            original_test_mode = self.test_mode
            self.test_mode = False
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = pipeline(input_dict)

        # extract data items according to keys
        if isinstance(key, str):
            data = extract_result_dict(example, key)
        else:
            data = [extract_result_dict(example, k) for k in key]
        if load_annos:
            self.test_mode = original_test_mode

        return data