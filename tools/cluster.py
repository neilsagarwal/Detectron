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
from subprocess import Popen
import shutil
from tqdm import tqdm
from termcolor import colored

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
from detectron.utils.tracker import ObjectTracker, TrackingMethod

import detectron.utils.env as envu
envu.set_up_matplotlib()
import matplotlib.pyplot as plt

import json

import numpy as np
sys.path.append("/home/ubuntu/sort")
# import sort

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


DEFAULT_CONFIG = '/home/ubuntu/detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml'
DEFAULT_IMG_EXT = 'jpg'
DEFAULT_WTS = 'https://dl.fbaipublicfiles.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl'
DEFAULT_OUT_EXT = 'jpg'
DEFAULT_THRESH = 0.5
DEFAULT_PRIVACY_THRESH = 0.5
DEFAULT_KP_THRESH = 2.0
PROFILE = False
DEBUG = False



def error(msg):
    sys.stderr.write("[{}] {}\n".format(colored('erro', 'red'), msg))
    sys.exit(1)
def warn(msg):
    sys.stderr.write("[{}] {}\n".format(colored('warn', 'yellow'), msg))
    sys.stdout.flush()
def info(msg):
    sys.stdout.write("[{}] {}\n".format(colored('info', 'green'), msg))
    sys.stdout.flush()


TMP_DIR = '/tmp/track'
tracking_algs = ['sort', 'centroid', 'iou']

workspace.GlobalInit(['caffe2', '--caffe2_log_level=1'])
setup_logging(__name__)
ARG_VIDEO = "/streams/goodstream/08192019-2139+3.mp4"
ARG_LEN = "00:00:30.0"
ARG_SAMPLE_RATE = 4
ARG_TRACK_CLASS = "person"
ARG_REBUILD_RATE = 4
ARG_NICK = "cluster"
ARG_TRACKER = "centroid"
ARG_CFG = DEFAULT_CONFIG
ARG_WEIGHTS = DEFAULT_WTS
ARG_IMG_EXT = DEFAULT_IMG_EXT
ARG_OUT_EXT = DEFAULT_OUT_EXT
ARG_PRIV_THRESH = DEFAULT_PRIVACY_THRESH

logger = logging.getLogger(__name__)

if not ARG_TRACKER in tracking_algs:
    logger.error('Unknown tracking algorithm. Available algorithms are: {}'.format(','.join(tracking_algs)))
    sys.exit(1)

merge_cfg_from_file(ARG_CFG)
cfg.NUM_GPUS = 1

ARG_WEIGHTS = cache_url(ARG_WEIGHTS, cfg.DOWNLOAD_CACHE)

assert_and_infer_cfg(cache_urls=False)

assert not cfg.MODEL.RPN_ONLY, \
    'RPN models are not supported'
assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
    'Models that require precomputed proposals are not supported'

model = infer_engine.initialize_model_from_cfg(ARG_WEIGHTS)

dummy_coco_dataset = dummy_datasets.get_coco_dataset()

path_to_video = ARG_VIDEO

nickname = '.'.join(path_to_video.split("/")[-1].split(".")[:-1])
if ARG_NICK:
    nickname = ARG_NICK

video_dir = os.path.join(TMP_DIR, nickname)
# if os.path.exists(video_dir):
#     shutil.rmtree(video_dir)
# os.makedirs(video_dir)

clipped_video = ARG_VIDEO

# info("Sampling video...")
# ffmpeg = Popen("ffmpeg -i {} -vf fps={} {} -hide_banner -loglevel error".format(
#     os.path.basename(clipped_video),
#     ARG_SAMPLE_RATE,
#     os.path.join(video_dir, "raw_%04d.jpg")
# ), shell=True, cwd=os.path.dirname(clipped_video))
#
# ffmpeg.wait()

im_list = sorted(glob.iglob(os.path.join(video_dir, '*.' + ARG_IMG_EXT)))

num_frames = len(im_list)
h,m,s = ARG_LEN.split(":")
total_length_seconds = float(h)*3600 + float(m) * 60 + float(s)
frame_length = total_length_seconds / num_frames
info("Created {} frames, each represent {:.3f} seconds".format(len(im_list), frame_length))

results_json = {}

counter = 0
info("Processing frames...")
fr_counter = 0
for im_name in tqdm(im_list, unit="frame", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}]"):
    frame_num = os.path.basename(im_name).split(".")[0].split("_")[1]
    fr_counter += 1

    im = cv2.imread(im_name)
    timers = defaultdict(Timer)
    t1 = time.time()
    with c2_utils.NamedCudaScope(0):
        original_boxes, original_segms, original_kps = infer_engine.im_detect_all(
            model, im, None, timers=timers
        )
    t2 = time.time()

    if isinstance(original_boxes, list):
        original_boxes, original_segms, original_kps, original_classes = vis_utils.convert_from_cls_format(
                original_boxes, original_segms, original_kps)
    if original_boxes is None or original_boxes.shape[0] == 0 or max(original_boxes[:, 4]) < ARG_PRIV_THRESH:
        logger.warn("No bounding boxes found!")
        continue

    for i in range(len(original_classes)):
        cls = dummy_coco_dataset.classes[original_classes[i]]
        if cls in ARG_TRACK_CLASS and original_boxes[i, -1] >= ARG_PRIV_THRESH:
            box = [original_boxes[i].astype(int)][0]
            # print(im)
            dpi=200
            plt.clf()
            fig = plt.figure(frameon=False)
            fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.axis('off')
            fig.add_axes(ax)
            print(im.shape)
            ax.imshow(im[box[1]:box[3], box[0]:box[2], :])

            # ax.add_patch(
            #     plt.Rectangle((box[0], box[1]),
            #     box[2] - box[0],
            #     box[3] - box[1],
            #     fill=True, edgecolor="#e6194b",
            #     linewidth=1.0, alpha=0.5))
            fig.savefig("/home/ubuntu/photos/test%s_%s.jpeg" % (fr_counter, counter), dpi=200)
            counter += 1
        # else:
        #     print("uhoh")
            # im[:, :, ::-1]
    # if counter > 400: break
