from collections import OrderedDict
from scipy.spatial.distance import cdist as pairwise_dist
import numpy as np
from enum import Enum

import sys
sys.path.append("/home/ubuntu/sort")
import sort

def find_centroid(box):
    x,y,w,h,_ = box
    x2 = x + w
    y2 = y + h
    return (int((x+x2)/2), int((y+y2)/2))
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
    if union <= 0:
        return 0
    iou = intersection / union
    return iou

class TrackingMethod(Enum):
    CENTROID = 1
    IOU = 2
    SORT = 3

class ObjectTracker(object):

    def __init__(self, missing_threshold=25, method=TrackingMethod.CENTROID, max_dist=100):
        self.next_id = 1
        self.name="{},missing={},max={}".format(str(method).split(".")[1], missing_threshold,max_dist)
        self.old_obj = OrderedDict()
        self.box_map = OrderedDict()
        self.last_seen = OrderedDict()
        self.missing_threshold = missing_threshold
        self.max_dist = max_dist
        self.method = method
        if self.method == TrackingMethod.CENTROID:
            self.metric = 'euclidean'
            self.update = self.update_dist
        elif self.method == TrackingMethod.IOU:
            self.metric = iou
            self.update = self.update_dist
        elif self.method == TrackingMethod.SORT:
            self.sort = sort.Sort()
            self.update = self.update_sort
        self.watching = {}
        
            
    def update_sort(self, boxes, frame_num):
        ids = self.sort.update(boxes).astype(int)
        self.watch(ids, frame_num)
        return ids


    def update_dist(self, boxes, frame_num):

        if len(boxes) == 0:
            to_forget = []
            for obj_id in self.last_seen.keys():
                self.last_seen[obj_id] += 1
                #self.box_map[obj_id] = None
                if self.last_seen[obj_id] > self.missing_threshold:
                    to_forget.append(obj_id)
            for obj_id in to_forget:
                self.forget(obj_id)
            #return self.box_map
            return boxes

        if self.method == TrackingMethod.CENTROID:
            new_obj = np.zeros((len(boxes), 2), dtype="int")
            for i in range(len(boxes)):
                new_obj[i] = find_centroid(boxes[i])
        elif self.method == TrackingMethod.IOU:
            #new_obj = np.zeros((len(boxes), 5), dtype="int")
            new_obj = np.copy(boxes)

        if len(self.old_obj) == 0:
            for i in range(0, len(new_obj)):
                uid = self.register(new_obj[i], boxes[i])
                boxes[i, -1] = uid
        else:

            old_ids = list(self.old_obj.keys())
            old_obj = np.array(list(self.old_obj.values()))

            D = pairwise_dist(old_obj, new_obj, metric=self.metric)

            old_matched = set()
            new_matched = set()

            if self.method == TrackingMethod.CENTROID:
                old_ind = D.min(axis=1).argsort()
            elif self.method == TrackingMethod.IOU:
                old_ind = D.max(axis=1).argsort()
            if self.method == TrackingMethod.CENTROID:
                new_ind = D.argmin(axis=1)[old_ind]
            elif self.method == TrackingMethod.IOU:
                new_ind = D.argmax(axis=1)[old_ind]

            for (oi, ni) in zip(old_ind, new_ind):
                if oi in old_matched or ni in new_matched:
                    continue
                if D[oi][ni] > self.max_dist:
                    continue
                obj_id = old_ids[oi]
                self.old_obj[obj_id] = new_obj[ni]
                #self.box_map[obj_id] = boxes[ni]
                boxes[ni][-1] = obj_id
                self.last_seen[obj_id] = 0
                old_matched.add(oi)
                new_matched.add(ni)

            # potentially disappeared
            old_not_matched = set(range(0, D.shape[0])).difference(old_matched)
            # potentially new object
            new_not_matched = set(range(0, D.shape[1])).difference(new_matched)

            if D.shape[0] >= D.shape[1]:
                for oi in old_not_matched:
                    obj_id = old_ids[oi]
                    self.last_seen[obj_id] += 1
                    #self.box_map[obj_id] = None
                    if self.last_seen[obj_id] > self.missing_threshold:
                        self.forget(obj_id)
            else:
                for ni in new_not_matched:
                    uid = self.register(new_obj[ni], boxes[ni])
                    boxes[ni, -1] = uid

        self.watch(boxes, frame_num)
        return boxes

    def watch(self, boxes, frame_num):
        for box in boxes:
            uid = str(box[-1])
            if uid in self.watching:
                self.watching[uid][1] = frame_num
            else:
                self.watching[uid] = [frame_num, frame_num]

    def write_watch_results(self):
        import json
        with open('./watch_results', 'w') as f:
            f.write(json.dumps(self.watching))

    def get_watch_results(self):
        return self.watching

    def get_name(self):
        return self.name

    def register(self, center, box):
        uid = self.next_id
        self.old_obj[uid] = center
        self.box_map[uid] = box
        self.last_seen[uid] = 0
        self.next_id += 1
        return uid

    def forget(self, obj_id):
        del self.old_obj[obj_id]
        del self.box_map[obj_id]
        del self.last_seen[obj_id]

