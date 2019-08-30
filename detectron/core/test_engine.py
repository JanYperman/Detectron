# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Test a Detectron network on an imdb (image database)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import datetime
import glob # Debug
import logging
import numpy as np
import os
import pandas
import pdb

from caffe2.python import workspace

from detectron.core.config import cfg
from detectron.core.config import get_output_dir
from detectron.core.rpn_generator import generate_rpn_on_dataset
from detectron.core.rpn_generator import generate_rpn_on_range
from detectron.core.test import im_detect_all, im_detect_all_batch
from detectron.datasets import task_evaluation
from detectron.datasets.video_dataset import Dataset_from_videos
from detectron.datasets.json_dataset import JsonDataset
from detectron.modeling import model_builder
from detectron.utils.io import save_object
from detectron.utils.timer import Timer
import detectron.utils.c2 as c2_utils
import detectron.utils.env as envu
import detectron.utils.net as net_utils
import detectron.utils.subprocess as subprocess_utils
import detectron.utils.vis as vis_utils

logger = logging.getLogger(__name__)


def get_eval_functions():
    # Determine which parent or child function should handle inference
    if cfg.MODEL.RPN_ONLY:
        child_func = generate_rpn_on_range
        parent_func = generate_rpn_on_dataset
    else:
        # Generic case that handles all network types other than RPN-only nets
        # and RetinaNet
        child_func = test_net
        parent_func = test_net_on_dataset

    return parent_func, child_func


def get_inference_dataset(index, is_parent=True):
    assert is_parent or len(cfg.TEST.DATASETS) == 1, \
        'The child inference process can only work on a single dataset'

    dataset_name = cfg.TEST.DATASETS[index]

    if cfg.TEST.PRECOMPUTED_PROPOSALS:
        assert is_parent or len(cfg.TEST.PROPOSAL_FILES) == 1, \
            'The child inference process can only work on a single proposal file'
        assert len(cfg.TEST.PROPOSAL_FILES) == len(cfg.TEST.DATASETS), \
            'If proposals are used, one proposal file must be specified for ' \
            'each dataset'
        proposal_file = cfg.TEST.PROPOSAL_FILES[index]
    else:
        proposal_file = None

    return dataset_name, proposal_file


def run_inference(
    weights_file, video_list, ind_range=None,
    multi_gpu_testing=False, gpu_id=0,
    check_expected_results=False
):
    parent_func, child_func = get_eval_functions()
    is_parent = ind_range is None

    def result_getter():
        if is_parent:
            # Parent case:
            # In this case we're either running inference on the entire dataset in a
            # single process or (if multi_gpu_testing is True) using this process to
            # launch subprocesses that each run inference on a range of the dataset
            #t all_results = {}
            for i in range(len(cfg.TEST.DATASETS)):
                dataset_name, proposal_file = get_inference_dataset(i)
                output_dir = get_output_dir(dataset_name, training=False)
                results = parent_func(
                    weights_file,
                    dataset_name,
                    proposal_file,
                    output_dir,
                    multi_gpu=multi_gpu_testing,
                    video_list=video_list
                )
                #t all_results.update(results)

            #t return all_results
        else:
            # Subprocess child case:
            # In this case test_net was called via subprocess.Popen to execute on a
            # range of inputs on a single dataset
            dataset_name, proposal_file = get_inference_dataset(0, is_parent=False)
            output_dir = get_output_dir(dataset_name, training=False)
            return child_func(
                weights_file,
                dataset_name,
                proposal_file,
                output_dir,
                ind_range=ind_range,
                gpu_id=gpu_id,
                video_list=video_list
            )

    all_results = result_getter()
    if check_expected_results and is_parent:
        task_evaluation.check_expected_results(
            all_results,
            atol=cfg.EXPECTED_RESULTS_ATOL,
            rtol=cfg.EXPECTED_RESULTS_RTOL
        )
        task_evaluation.log_copy_paste_friendly_results(all_results)

    return all_results



def test_net_on_dataset(
    weights_file,
    dataset_name,
    proposal_file,
    output_dir,
    multi_gpu=False,
    gpu_id=0,
    video_list=None
):
    """Run inference on a dataset."""
    dataset = Dataset_from_videos(video_list)
    # dataset = JsonDataset(dataset_name)
    test_timer = Timer()
    test_timer.tic()
    if multi_gpu:
        #debug num_images = len(dataset.get_roidb())
        # num_images = 5
        num_images = dataset.tot_frames
        all_boxes, all_segms, all_keyps = multi_gpu_test_net_on_dataset(
            weights_file, dataset_name, proposal_file, num_images, output_dir,
            video_list=video_list
        )
    else:
        all_boxes, all_segms, all_keyps = test_net(
            weights_file, dataset_name, proposal_file, output_dir, gpu_id=gpu_id,
            video_list=video_list
        )
    test_timer.toc()
    logger.info('Total inference time: {:.3f}s'.format(test_timer.average_time))
    # results = task_evaluation.evaluate_all(
    #     dataset, all_boxes, all_segms, all_keyps, output_dir
    # )
    # return results
    return None


def multi_gpu_test_net_on_dataset(
    weights_file, dataset_name, proposal_file, num_images, output_dir, video_list
):
    """Multi-gpu inference on a dataset."""
    binary_dir = envu.get_runtime_dir()
    binary_ext = envu.get_py_bin_ext()
    binary = os.path.join(binary_dir, 'test_net' + binary_ext)
    assert os.path.exists(binary), 'Binary \'{}\' not found'.format(binary)

    # Pass the target dataset and proposal file (if any) via the command line
    opts = ['TEST.DATASETS', '("{}",)'.format(dataset_name)]
    opts += ['TEST.WEIGHTS', weights_file]
    if proposal_file:
        opts += ['TEST.PROPOSAL_FILES', '("{}",)'.format(proposal_file)]

    # Run inference in parallel in subprocesses
    # Outputs will be a list of outputs from each subprocess, where the output
    # of each subprocess is the dictionary saved by test_net().
    outputs = subprocess_utils.process_in_parallel(
        'detection', num_images, binary, output_dir, opts, video_list
    )

    ## # Collate the results from each subprocess
    ## all_boxes = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    ## all_segms = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    ## all_keyps = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    ## for det_data in outputs:
    ##     all_boxes_batch = det_data['all_boxes']
    ##     all_segms_batch = det_data['all_segms']
    ##     all_keyps_batch = det_data['all_keyps']
    ##     for cls_idx in range(1, cfg.MODEL.NUM_CLASSES):
    ##         all_boxes[cls_idx] += all_boxes_batch[cls_idx]
    ##         all_segms[cls_idx] += all_segms_batch[cls_idx]
    ##         all_keyps[cls_idx] += all_keyps_batch[cls_idx]
    ## det_file = os.path.join(output_dir, 'detections.pkl')
    ## cfg_yaml = envu.yaml_dump(cfg)
    ## save_object(
    ##     dict(
    ##         all_boxes=all_boxes,
    ##         all_segms=all_segms,
    ##         all_keyps=all_keyps,
    ##         cfg=cfg_yaml
    ##     ), det_file
    ## )
    det_file = '%s_dets.pkl' % (os.path.splitext(video_list)[0])
    logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))

    return None, None, None
    # return all_boxes, all_segms, all_keyps



def test_net(
    weights_file,
    dataset_name,
    proposal_file,
    output_dir,
    ind_range=None,
    gpu_id=0,
    video_list=None
):
    """Run inference on all images in a dataset or over an index range of images
    in a dataset using a single GPU.
    """
    assert not cfg.MODEL.RPN_ONLY, \
        'Use rpn_generate to generate proposals from RPN-only models'

    # if cfg.TEST.IMS_PER_BATCH != 1:
    return test_net_batch(weights_file, dataset_name, proposal_file, output_dir, ind_range, gpu_id, video_list)

@profile
def test_net_batch(
    weights_file,
    dataset_name,
    proposal_file,
    output_dir,
    ind_range=None,
    gpu_id=0,
    video_list=None
):
    """Run inference on all images in a dataset or over an index range of images
    in a dataset using a single GPU, using batch inference
    """
    assert not cfg.MODEL.RPN_ONLY, \
        'Use rpn_generate to generate proposals from RPN-only models'

    _, dataset, start_ind, end_ind, total_num_images = get_roidb_and_dataset(
        dataset_name, proposal_file, ind_range, video_list
    )

    # Skip to starting frame
    dataset.skip_to_frame(start_ind)

    # roidb = roidb[:500]
    # Debug purposes
    # roidb = [{'image': im} for im in glob.glob('/staging/leuven/stg_00027/imob/detectron/lib/datasets/data/coco/coco_val2014/*.png')][:500]

    model = initialize_model_from_cfg(weights_file, gpu_id=gpu_id)
    if ind_range is not None:
        num_images = ind_range[-1] - ind_range[0]
    else:
        num_images = dataset.tot_frames
    # num_images = len(roidb)
    num_classes = cfg.MODEL.NUM_CLASSES
    # all_boxes, all_segms, all_keyps = empty_results(num_classes, num_images)

    if cfg.TEST.PRECOMPUTED_PROPOSALS:
        raise NotImplementedError('Precomputed proposals not implemented for batch inference, set TEST.IMS_PER_BATCH to 1')
    else:
        # Faster R-CNN type models generate proposals on-the-fly with an
        # in-network RPN; 1-stage models don't require proposals.
        box_proposals = None

    timers = defaultdict(Timer)

    # We'll create a dataframe from list of dicts
    dets_df = []

    ims_next = []
    metadata_batch_next = []
    cursor = 0
    while cursor < num_images:
        timers['per_frame'].tic()
        logger.info('cursor: %i, av_per_frame: %.2f' % (cursor, timers['per_frame'].average_time / cfg.TEST.IMS_PER_BATCH))
        ims = []
        metadata_batch = []

        if len(ims_next) == 0:
            # Get a frame
            im, metadata = dataset.get_next_frame()
            cursor += 1
        else:
            assert len(ims_next) == 1
            im = ims_next[0]
            metadata = metadata_batch_next[0]
        ims_next = []
        metadata_batch_next = []
        ims.append(im)
        metadata_batch.append(metadata)
        
        im_size = im.shape[0] * im.shape[1]
        batch_size = cfg.TEST.IMS_PER_BATCH
        
        # Fill batch list until batch_size or image doesn't have the same size
        for i in range(batch_size - 1):
            if not (cursor < num_images):
                break
            im, metadata = dataset.get_next_frame()
            cursor += 1
            # In case the size of the next image is different (may not fit in memory)
            if im.shape[0] * im.shape[1] != im_size:
                ims_next.append(im)
                metadata_batch_next.append(metadata)
                break
            ims.append(im)
            metadata_batch.append(metadata)

        with c2_utils.NamedCudaScope(gpu_id):
            cls_boxes_batch, cls_segms_batch, cls_keyps_batch = im_detect_all_batch(
                model, ims, box_proposals, timers
            )

        for n in range(len(ims)):
            cls_boxes_i = cls_boxes_batch[n]
            cls_segms_i = cls_segms_batch[n] if cls_segms_batch else None
            cls_keyps_i = cls_keyps_batch[n] if cls_keyps_batch else None

            for cls in range(1, len(cls_boxes_i)):
                for det_idx in range(len(cls_boxes_i[cls])):
                    dets_df.append({
                        'class': cls,
                        'boxes': cls_boxes_i[cls][det_idx],
                        'segms': cls_segms_i[cls][det_idx]
                        })
                # Add the video metadata
                dets_df[-1].update(metadata_batch[n])

            ## extend_results(local_i, all_boxes, cls_boxes_i)
            ## if cls_segms_i is not None:
            ##     extend_results(local_i, all_segms, cls_segms_i)
            ## if cls_keyps_i is not None:
            ##     extend_results(local_i, all_keyps, cls_keyps_i)

            ## if local_i % (10 * cfg.TEST.IMS_PER_BATCH) == 0:  # Reduce log file size
            ##     ave_total_time = np.sum([t.average_time for t in timers.values()])
            ##     eta_seconds = int(ave_total_time * (num_images - local_i - 1) / float(cfg.TEST.IMS_PER_BATCH))
            ##     eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            ##     det_time = (
            ##         timers['im_detect_bbox'].average_time +
            ##         timers['im_detect_mask'].average_time +
            ##         timers['im_detect_keypoints'].average_time
            ##     )
            ##     misc_time = (
            ##         timers['misc_bbox'].average_time +
            ##         timers['misc_mask'].average_time +
            ##         timers['misc_keypoints'].average_time
            ##     )
            ##     logger.info(
            ##         (
            ##             'im_detect: range [{:d}, {:d}] of {:d}: '
            ##             '{:d}/{:d} {:.3f}s + {:.3f}s (eta: {})'
            ##         ).format(
            ##             start_ind + 1, end_ind, total_num_images, start_ind + local_i + 1,
            ##             start_ind + num_images, det_time, misc_time, eta
            ##         )
            ##     )

            ## # This will now only show the last image of each batch
            ## if cfg.VIS:
            ##     im_name = os.path.splitext(os.path.basename(entry['image']))[0]
            ##     vis_utils.vis_one_image(
            ##         im[:, :, ::-1],
            ##         '{:d}_{:s}'.format(i, im_name),
            ##         os.path.join(output_dir, 'vis'),
            ##         cls_boxes_i,
            ##         segms=cls_segms_i,
            ##         keypoints=cls_keyps_i,
            ##         thresh=cfg.VIS_TH,
            ##         box_alpha=0.8,
            ##         dataset=dataset,
            ##         show_class=True
            ##     )
        ## ims = []
        timers['per_frame'].toc()

    cfg_yaml = envu.yaml_dump(cfg)
    df = pd.DataFrame(dets_df)
    if ind_range is not None:
        det_name = 'detection_range_%s_%s.pkl' % tuple(ind_range)
    else:
        det_name = 'detections.pkl'
    det_file = os.path.join(output_dir, det_name)
    df.to_pickle(det_file)
    ## save_object(
    ##     dict(
    ##         all_boxes=all_boxes,
    ##         all_segms=all_segms,
    ##         all_keyps=all_keyps,
    ##         cfg=cfg_yaml
    ##     ), det_file
    ## )
    logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))
    return None, None, None
    # return all_boxes, all_segms, all_keyps


def initialize_model_from_cfg(weights_file, gpu_id=0):
    """Initialize a model from the global cfg. Loads test-time weights and
    creates the networks in the Caffe2 workspace.
    """
    model = model_builder.create(cfg.MODEL.TYPE, train=False, gpu_id=gpu_id)
    net_utils.initialize_gpu_from_weights_file(
        model, weights_file, gpu_id=gpu_id,
    )
    model_builder.add_inference_inputs(model)
    workspace.CreateNet(model.net)
    workspace.CreateNet(model.conv_body_net)
    if cfg.MODEL.MASK_ON:
        workspace.CreateNet(model.mask_net)
    if cfg.MODEL.KEYPOINTS_ON:
        workspace.CreateNet(model.keypoint_net)
    return model


def get_roidb_and_dataset(dataset_name, proposal_file, ind_range, video_list):
    """Get the roidb for the dataset specified in the global cfg. Optionally
    restrict it to a range of indices if ind_range is a pair of integers.
    """
    # dataset = JsonDataset(dataset_name)
    dataset = Dataset_from_videos(video_list)
    ## if cfg.TEST.PRECOMPUTED_PROPOSALS:
    ##     assert proposal_file, 'No proposal file given'
    ##     roidb = dataset.get_roidb(
    ##         proposal_file=proposal_file,
    ##         proposal_limit=cfg.TEST.PROPOSAL_LIMIT
    ##     )
    ## else:
    ## roidb = dataset.get_roidb()

    if ind_range is not None:
        ## total_num_images = len(roidb)
        total_num_images = dataset.tot_frames
        start, end = ind_range
        ## roidb = roidb[start:end]
    else:
        start = 0
        ## end = len(roidb)
        end = dataset.tot_frames
        total_num_images = end

    return None, dataset, start, end, total_num_images


def empty_results(num_classes, num_images):
    """Return empty results lists for boxes, masks, and keypoints.
    Box detections are collected into:
      all_boxes[cls][image] = N x 5 array with columns (x1, y1, x2, y2, score)
    Instance mask predictions are collected into:
      all_segms[cls][image] = [...] list of COCO RLE encoded masks that are in
      1:1 correspondence with the boxes in all_boxes[cls][image]
    Keypoint predictions are collected into:
      all_keyps[cls][image] = [...] list of keypoints results, each encoded as
      a 3D array (#rois, 4, #keypoints) with the 4 rows corresponding to
      [x, y, logit, prob] (See: utils.keypoints.heatmaps_to_keypoints).
      Keypoints are recorded for person (cls = 1); they are in 1:1
      correspondence with the boxes in all_boxes[cls][image].
    """
    # Note: do not be tempted to use [[] * N], which gives N references to the
    # *same* empty list.
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_segms = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_keyps = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    return all_boxes, all_segms, all_keyps


def extend_results(index, all_res, im_res):
    """Add results for an image to the set of all results at the specified
    index.
    """
    # Skip cls_idx 0 (__background__)
    for cls_idx in range(1, len(im_res)):
        #try:
        all_res[cls_idx][index] = im_res[cls_idx]
        #except:
            #pdb.set_trace()
