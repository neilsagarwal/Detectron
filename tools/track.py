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
from detectron.utils.tracker import ObjectTracker

import detectron.utils.env as envu
envu.set_up_matplotlib()
import matplotlib.pyplot as plt

import json

import numpy as np
sys.path.append("/home/ubuntu/sort")
import sort

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

def parse_args():
    parser = argparse.ArgumentParser(description='Track unique objets in video')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=DEFAULT_CONFIG,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=DEFAULT_WTS,
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
        default=DEFAULT_IMG_EXT,
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
        default=DEFAULT_OUT_EXT,
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='Threshold for visualizing detections',
        default=DEFAULT_THRESH,
        type=float
    )
    parser.add_argument(
        '--privacy-thresh',
        dest='privacy_thresh',
        help='Threshold for detecting objects that should be denatured',
        default=DEFAULT_PRIVACY_THRESH,
        type=float
    )
    parser.add_argument(
        '--kp-thresh',
        dest='kp_thresh',
        help='Threshold for visualizing keypoints',
        default=DEFAULT_KP_THRESH,
        type=float
    )
    parser.add_argument(
        '--start',
        help="Beginning timestamp of video to use, format: 00:00:00.0"
    )
    parser.add_argument(
        '--length',
        help="Length of video clip, format: 00:00:00.0"
    )
    parser.add_argument(
        '--clip',
        help="If provided, clip the video using required --start and --length arguments",
        action='store_true'
    )
    parser.add_argument(
        '--sample-rate',
        help="How frequently to sample frames from the video",
        required=True
    )
    parser.add_argument(
        '--rebuild-rate',
        help="FPS of the final denatured video"
    )
    parser.add_argument(
        '--tracker',
        help="Which object tracking algorithm to use: (" + "|".join(tracking_algs) + ")"
    )
    parser.add_argument(
        '--track-class',
        help="Class of objects to track",
        type=(lambda s : s.replace('[','').replace(']','').split(",")),
        required=True
    )
    #parser.add_argument(
    #    '--query-class',
    #    help="List of classes of objects to query for, comma-separated",
    #    required=True,
    #    type=(lambda s : s.replace('[','').replace(']','').split(","))
    #)
    parser.add_argument(
        '--nickname',
        help="Name for this run, stored in /tmp/track",
        required=True
    )
    parser.add_argument(
        'video', help='video', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

TMP_DIR = '/tmp/track'
tracking_algs = ['sort', 'centroid', 'iou']

def main(args):
    logger = logging.getLogger(__name__)

    if not args.tracker in tracking_algs:
        logger.error('Unknown tracking algorithm. Available algorithms are: {}'.format(','.join(tracking_algs)))
        sys.exit(1)

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

    path_to_video = args.video

    nickname = '.'.join(path_to_video.split("/")[-1].split(".")[:-1])
    if args.nickname:
        nickname = args.nickname

    video_dir = os.path.join(TMP_DIR, nickname)
    if os.path.exists(video_dir):
        shutil.rmtree(video_dir)
    os.makedirs(video_dir)

    clipped_video = os.path.join(video_dir, "clipped.mp4")


    if args.clip:
        if not 'start' in args or not 'length' in args:
            logger.error("If --clip is provided, --start and --length must also be provided")
            sys.exit(1)
        if len(args.start.split(":")) != 3:
            logger.error("--start must be of the format 00:00:00")
            sys.exit(1)
        if len(args.length.split(":")) != 3:
            logger.error("--length must be of the format 00:00:00.0")
            sys.exit(1)

        info("Clipping video...")

        ffmpeg = Popen("ffmpeg -hide_banner -loglevel error -ss {} -i {} -c copy -t {} {}".format(
            args.start,
            args.video,
            args.length,
            clipped_video
        ), shell=True)
        ffmpeg.wait()
    else:
        clipped_video = args.video

    info("Sampling video...")
    ffmpeg = Popen("ffmpeg -i {} -vf fps={} {} -hide_banner -loglevel error".format(
        os.path.basename(clipped_video),
        args.sample_rate,
        os.path.join(video_dir, "raw_%04d.jpg")
    ), shell=True, cwd=os.path.dirname(clipped_video))

    ffmpeg.wait()

    im_list = sorted(glob.iglob(os.path.join(video_dir, '*.' + args.image_ext)))

    num_frames = len(im_list)
    h,m,s = args.length.split(":")
    total_length_seconds = float(h)*3600 + float(m) * 60 + float(s)
    frame_length = total_length_seconds / num_frames
    info("Created {} frames, each represent {:.3f} seconds".format(len(im_list), frame_length))

    results_json = {}


    if args.tracker == 'centroid':
        tracker = ObjectTracker(method='centroid')
    elif args.tracker == 'iou':
        tracker = ObjectTracker(method='iou')
    elif args.tracker == 'sort':
        tracker = sort.Sort()

    # Object Tracking state
    ooi = {}
    seen = set()
    all_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
    id_to_color = {}
    color_i = 0

    #out = open('out.txt', 'w')
    info("Processing frames...")
    for im_name in tqdm(im_list, unit="frame", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}]"):

        frame_num = os.path.basename(im_name).split(".")[0].split("_")[1]

        im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        t1 = time.time()
        with c2_utils.NamedCudaScope(0):
            original_boxes, original_segms, original_kps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        t2 = time.time()
        if PROFILE:
            for t in timers.keys():
                print(t, timers[t].total_time, timers[t].average_time)
            print("Total inference time: ", t2-t1)


        if isinstance(original_boxes, list):
            original_boxes, original_segms, original_kps, original_classes = vis_utils.convert_from_cls_format(
                    original_boxes, original_segms, original_kps)
        if original_boxes is None or original_boxes.shape[0] == 0 or max(original_boxes[:, 4]) < args.privacy_thresh:
            logger.warn("No bounding boxes found!")
            continue
 
        boi = np.zeros((len(original_classes), 5), dtype=int)
        for i in range(len(original_classes)):
            cls = dummy_coco_dataset.classes[original_classes[i]]
            if cls in args.track_class and original_boxes[i, -1] >= args.thresh:
                boi[i] = original_boxes[i]
        
        if args.tracker == 'centroid':
            track_bbs_ids = tracker.update(boi)
            #ids_in_frame = set(box_map.keys())
        elif args.tracker == 'iou':
            track_bbs_ids = tracker.update(boi)
        elif args.tracker == 'sort':
            track_bbs_ids = mot_tracker.update(boi).astype(int)

        ids_in_frame = set(track_bbs_ids[:, -1])
        for uid in ids_in_frame:
            if not uid in id_to_color:
                id_to_color[uid] = all_colors[color_i]
                color_i = (color_i + 1) % len(all_colors)
        print("im_name", ids_in_frame)
        new_ids = seen.symmetric_difference(ids_in_frame)

        ids_fig = vis_utils.vis_sort(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            track_bbs_ids, 
            id_to_color,
            new_ids
        )
        seen.update(ids_in_frame)

        ids_frame_name = 'ids_' + frame_num
        ids_frame_path = os.path.join(video_dir, ids_frame_name + "." + args.output_ext)
        ids_fig.savefig(ids_frame_path, dpi=200)
        plt.close('all')


    info("Rebuilding video...")

    ffmpeg = Popen("ffmpeg -loglevel error -hide_banner -r {} -i {} {}".format(
        args.rebuild_rate,
        os.path.join(video_dir, "ids_%04d." + args.output_ext),
        "ids.mp4"
    ), shell=True, cwd=video_dir)

    ffmpeg.wait()

    info("Done.")

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=1'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
