#!/usr/bin/env python

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

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
from pprint import pprint

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--always-out',
        dest='out_when_no_box',
        help='output image even when no object is found',
        action='store_true'
    )
    parser.add_argument(
        '--output-ext',
        dest='output_ext',
        help='output image file format (default: pdf)',
        default='pdf',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='Threshold for visualizing detections',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--kp-thresh',
        dest='kp_thresh',
        help='Threshold for visualizing keypoints',
        default=2.0,
        type=float
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    logger = logging.getLogger(__name__)

    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)

    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'

    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    for i, im_name in enumerate(im_list):
        im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        #logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        #for k, v in timers.items():
        #    logger.info(' | {}: {:.3f}s'.format(k, v.average_time))

        raw_im_name = im_name.split("/")[-1]
        print("denaturing...")
        car_count_before, carbox_before, blackbox_before = vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            raw_im_name,
            '/tmp/blackout/before',
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=args.thresh,
            kp_thresh=args.kp_thresh,
            ext=args.output_ext,
            out_when_no_box=args.out_when_no_box,
            remove_class="person",
            #find_class=["car", "bus", "truck"],
            find_class=None,
            denaturing=vis_utils.Denaturing.MASK
        )

        tmp_loc = '/tmp/blackout/before/'+raw_im_name+'.'+args.output_ext
        im_after = cv2.imread(tmp_loc)
        timers = defaultdict(Timer)
        with c2_utils.NamedCudaScope(0):
            cls_boxes_after, cls_segms_after, cls_keyps_after = infer_engine.im_detect_all(
                    model, im_after, None, timers=timers
            )
        
        print("object detection on denatured...")
        car_count_after, carbox_after, blackbox_after = vis_utils.vis_one_image(
            im_after[:, :, ::-1],
            raw_im_name,
            '/tmp/blackout/after',
            cls_boxes_after,
            cls_segms_after,
            cls_keyps_after,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=args.thresh,
            kp_thresh=args.kp_thresh,
            ext=args.output_ext,
            out_when_no_box=args.out_when_no_box,
            remove_class=None,
            #find_class=["car", "bus", "truck"],
            find_class=None,
            denaturing=None
        )

        car_change = 0.0
        if car_count_before > 0:
            car_change = float(car_count_after) / car_count_before
        print ('\tvehicles {:.2f}% = {} -> {}'.format(car_change, car_count_before, car_count_after))

        
        before_centers = []
        for car in carbox_before:
            before_centers.append(comp_center(*car))
        after_centers = []
        for car in carbox_after:
            after_centers.append(comp_center(*car))

        missing = 0
        for (x1,y1) in before_centers:
            found = False
            for (x2,y2) in after_centers:
                if abs(x1-x2) < 10 and abs(y2-y1) < 10:
                    print("({},{}) = ({},{})".format(x1,y1,x2,y2))
                    found = True
            if not found:
                missing += 1
                find_closest_person(x1,y1,blackbox_before)

def find_closest_person(x,y,people):
    pass

def comp_center(x,y,w,h):
    x2 = x + w
    y2 = y + h
    return ((x+x2)/2, (y+y2)/2)
    

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
