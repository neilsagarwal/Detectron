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

import detectron.utils.env as envu
envu.set_up_matplotlib()
import matplotlib.pyplot as plt

import json

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
    parser = argparse.ArgumentParser(description='Denature video, then run through object classifier')
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
        '--denaturing',
        dest='denaturing',
        help="Denaturing mechanism: (BOX|MASK|KSAME|INPAINT)",
        type=(lambda val : vis_utils.Denaturing(val.upper()))
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
        '--denature-class',
        help="Class of objects to denature",
        type=(lambda s : s.replace('[','').replace(']','').split(",")),
        required=True
    )
    parser.add_argument(
        '--query-class',
        help="List of classes of objects to query for, comma-separated",
        required=True,
        type=(lambda s : s.replace('[','').replace(']','').split(","))
    )
    parser.add_argument(
        'video', help='video', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

TMP_DIR = '/tmp/denature'

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

    path_to_video = args.video
    video_name = '.'.join(path_to_video.split("/")[-1].split(".")[:-1])

    video_dir = os.path.join(TMP_DIR, video_name, args.denaturing.value.lower())
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
    info("Created {} frames.".format(len(im_list)))

    results = {'frames' : []}

    info("Processing frames...")
    for im_name in tqdm(im_list, unit="frame", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}]"):
        im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        t2 = time.time()
        print("Total inference time: ", t2-t)
        for t in timers.keys():
            print(t, timers[t].total_time, timers[t].average_time)

        frame_result = {}

        frame_num = os.path.basename(im_name).split(".")[0].split("_")[1]
        denatured_frame_name = 'denatured_' + frame_num
        t = time.time()
        before_counts, before_boxes = vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            denatured_frame_name,
            video_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=False,
            thresh=args.privacy_thresh,
            kp_thresh=args.kp_thresh,
            ext=args.output_ext,
            out_when_no_box=args.out_when_no_box,
            remove_class=args.denature_class,
            find_class=args.query_class,
            denaturing=args.denaturing
        )
        t2 = time.time()
        print("Total vis time: ", t2-t)
        denatured_frame_path = os.path.join(video_dir, denatured_frame_name + "." + args.output_ext)
        frame_result['before'] = {}
        frame_result['before']['counts'] = before_counts
        frame_result['before']['boxes'] = before_boxes


        im_after = cv2.imread(denatured_frame_path)
        timers = defaultdict(Timer)
        with c2_utils.NamedCudaScope(0):
            cls_boxes_after, cls_segms_after, cls_keyps_after = infer_engine.im_detect_all(
                    model, im_after, None, timers=timers
            )

        def class_to_color(cls):
            if cls in args.denature_class:
                return 6
            elif cls in args.query_class:
                return 2
            else:
                return 3
            
        final_frame_name = 'final_' + frame_num
        fig, output_path, dpi, after_counts, after_boxes = vis_utils.vis_one_image(
            im_after[:, :, ::-1],
            final_frame_name,
            video_dir,
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
            find_class=args.query_class,
            denaturing=None,
            save=False
        )
        frame_result['after'] = {}
        frame_result['after']['counts'] = after_counts
        frame_result['after']['boxes'] = after_boxes

        results['frames'].append(frame_result)

        for cls in before_counts:
            before = before_counts[cls]
            after = after_counts[cls]
            change = int((float(after) / before) * 100) if before > 0 else 0
            print("{} {}% {}->{}".format(cls, change, before, after))

        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.clear()
        ax.axis('off')

        matched = [False for i in range(len(before_boxes))]
        for (found_box, found_cls) in after_boxes:
            max_so_far = 0.0
            max_cls = None
            max_i = -1
            for (i, (original_box, original_cls)) in enumerate(before_boxes):
                dist = iou(found_box, original_box)
                #print("\t", dist, original_box, original_cls)
                if dist > max_so_far:
                    max_so_far = dist
                    max_cls = original_cls
                    max_i = i 
            print("{} {} ".format(found_cls, found_box), end='')
            if max_so_far > 0.5:
                overlap = False
                if matched[max_i]:
                    overlap = True
                matched[max_i] = True
                if found_cls == max_cls:
                    print("true-positive {}".format("overlap" if overlap else ""))
                    draw_box(ax, found_box, "g", found_cls)
                else:
                    print("misclassified ({}) {}".format(max_cls, "overlap" if overlap else ""))
                    draw_box(ax, found_box, "purple", "f={},o={}".format(found_cls, max_cls))
            else:
                print("false-positive")
                draw_box(ax, found_box, "orange", found_cls)

        for i in range(len(matched)):
            if not matched[i]:
                box, cls = before_boxes[i]
                print("{} {} false-negative".format(cls, box))
                draw_box(ax, box, "red", cls)

        fig.add_axes(ax)




        #acc_ax = plt.Axes(fig, [0.8, 0.85, 0.2, 0.08])
        #acc_ax.clear()
        #acc_ax.axis('off')

        #after_people = after_counts['person']
        #before_people = before_counts['person']
        #people_perc = int((float(after_people) / before_people) * 100) if before_people > 0 else 0
        #after_vehicle = after_counts['car'] + after_counts['bus'] + after_counts['truck'] + after_counts['taxi']
        #before_vehicle = before_counts['car'] + before_counts['bus'] + before_counts['truck'] + before_counts['taxi']
        #vehicle_perc = int((float(after_vehicle) / before_vehicle) * 100) if before_vehicle > 0 else 0

        #bars = acc_ax.barh([0,1], [1,1], alpha=0.8, linewidth=1)
        #bars[0].set_color(acc_map(people_perc))
        #bars[1].set_color(acc_map(vehicle_perc))
        #acc_ax.text(0,bars[0].get_y()+0.35,'people', fontsize=5)
        #acc_ax.text(0,bars[0].get_y(),'truth={} diff={:+d}'.format(before_people, after_people-before_people), fontsize=5)
        #acc_ax.text(0,bars[1].get_y()+0.35,'vehicles', fontsize=5)
        #acc_ax.text(0,bars[1].get_y(),'truth={} diff={:+d}'.format(before_vehicle, after_vehicle-before_vehicle), fontsize=5)
        #acc_ax.set_xlim(left=0, right=1)
        #fig.add_axes(acc_ax)

        fig.savefig(output_path, dpi=dpi)

        final_frame_path = os.path.join(video_dir, final_frame_name + "." + args.output_ext)

        
        #before_centers = []
        #for car in carbox_before:
        #    before_centers.append(comp_center(*car))
        #after_centers = []
        #for car in carbox_after:
        #    after_centers.append(comp_center(*car))

        #print(car_count_before, car_count_after)
        #missing = 0
        #for (x1,y1) in before_centers:
        #    found = False
        #    for (x2,y2) in after_centers:
        #        if abs(x1-x2) < 2 and abs(y2-y1) < 2:
        #            #print("({},{}) = ({},{})".format(x1,y1,x2,y2))
        #            found = True
        #    if not found:
        #        missing += 1
        #        print (x1,y1)
        #        #find_closest_person(x1,y1,blackbox_before)

    info("Rebuilding video...")

    ffmpeg = Popen("ffmpeg -loglevel error -hide_banner -r {} -i {} {}".format(
        args.rebuild_rate,
        os.path.join(video_dir, "denatured_%04d." + args.output_ext),
        "denatured.mp4"
    ), shell=True, cwd=video_dir)

    ffmpeg.wait()

    ffmpeg = Popen("ffmpeg -loglevel error -hide_banner -r {} -i {} {}".format(
        args.rebuild_rate,
        os.path.join(video_dir, "final_%04d." + args.output_ext),
        "final.mp4"
    ), shell=True, cwd=video_dir)

    ffmpeg.wait()

    with open(os.path.join(video_dir, 'results.json'),'w') as f:
        f.write(json.dumps(results))

    info("Done.")

def acc_map(val):
    if val == 100:
        return '#2ecc71'
    elif val >= 80:
        return '#f1c40f'
    else:
        return '#e74c3c'
    
def comp_center(x,y,w,h):
    x2 = x + w
    y2 = y + h
    return ((x+x2)/2, (y+y2)/2)

def iou(a,b):
    x_left = max(a[0], b[0])
    x_right = min(a[2], b[2])
    y_bottom = max(a[1], b[1])
    y_top = min(a[3], b[3])
    if x_left > x_right or y_bottom > y_top:
        return 0.0
    intersection = float((x_right - x_left) * (y_top - y_bottom))
    a_area = (a[3] - a[1]) * (a[2] - a[0])
    b_area = (b[3] - b[1]) * (b[2] - b[0])
    union = float(a_area + b_area - intersection)
    iou = intersection / union
    return iou

def draw_box(ax, bbox, color, text):
    ax.add_patch(plt.Rectangle(
        (bbox[0], bbox[1]),
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
        fill=True,
        edgecolor=color,
        linewidth=1.5,
        alpha=0.5
    ))
    ax.text(bbox[0], bbox[1] - 2, text, fontsize=5, family='serif', color='white',
            bbox=dict(facecolor=color, alpha=0.4, pad=0, edgecolor='none'))


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=1'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
