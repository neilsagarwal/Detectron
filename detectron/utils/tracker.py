from collections import OrderedDict
from scipy.spatial.distance import cdist as pairwise_dist
import numpy as np

def find_centroid(box):
    x,y,w,h,_ = box
    x2 = x + w
    y2 = y + h
    return (int((x+x2)/2), int((y+y2)/2))

class ObjectTracker(object):
    MISSING_THRESHOLD = 25

    def __init__(self, method='centroid'):
        self.next_id = 1
        self.old_obj = OrderedDict()
        self.box_map = OrderedDict()
        self.last_seen = OrderedDict()
        self.method = method

    def update(self, boxes):

        if len(boxes) == 0:
            to_forget = []
            for obj_id in self.last_seen.keys():
                self.last_seen[obj_id] += 1
                #self.box_map[obj_id] = None
                if self.last_seen[obj_id] > ObjectTracker.MISSING_THRESHOLD:
                    to_forget.append(obj_id)
            for obj_id in to_forget:
                self.forget(obj_id)
            #return self.box_map
            return boxes

        new_obj = np.zeros((len(boxes), 2), dtype="int")
        for i in range(len(boxes)):
            new_obj[i] = find_centroid(boxes[i])

        if len(self.old_obj) == 0:
            for i in range(0, len(new_obj)):
                self.register(new_obj[i], boxes[i])
        else:

            old_ids = list(self.old_obj.keys())
            old_obj = np.array(list(self.old_obj.values()))

            D = pairwise_dist(old_obj, new_obj)

            old_matched = set()
            new_matched = set()

            old_ind = D.min(axis=1).argsort()
            new_ind = D.argmin(axis=1)[old_ind]
            for (oi, ni) in zip(old_ind, new_ind):
                if oi in old_matched or ni in new_matched:
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
                    if self.last_seen[obj_id] > ObjectTracker.MISSING_THRESHOLD:
                        self.forget(obj_id)
            else:
                for ni in new_not_matched:
                    self.register(new_obj[ni], boxes[ni])

        #return self.box_map
        return boxes

    def register(self, center, box):
        self.old_obj[self.next_id] = center
        self.box_map[self.next_id] = box
        self.last_seen[self.next_id] = 0
        self.next_id += 1

    def forget(self, obj_id):
        del self.old_obj[obj_id]
        del self.box_map[obj_id]
        del self.last_seen[obj_id]

