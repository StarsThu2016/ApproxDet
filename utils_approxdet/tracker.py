import numpy as np
from collections import defaultdict
import cv2 as cv

def import_detection_file(detection_file):

    # detection maps "name_of_image" to "list of bboxes" 
    #  -- (cls, conf, ymin, xmin, ymax, xmax)
    detection = defaultdict(lambda: [])

    # File format: we assume the detection file to have the following layout
    # name_of_image  class_id  confidence  (ymin,xmin,ymax,xmax)
    # xxxx.jpg       1         0.995         10    200  30   400
    with open(detection_file) as f:
        lines = f.readlines()
    # Extract the detection for each class
    for line in lines:
        items = line.strip().split()
        name, cls, conf = items[0], int(items[1]), float(items[2])
        ymin, xmin = float(items[3]), float(items[4])
        ymax, xmax = float(items[5]), float(items[6])
        detection[name].append((cls, conf, ymin, xmin, ymax, xmax))
    return detection

class OpenCVTracker:

    def __init__(self, ds = 1, name = 'kcf'):

        self.ds = ds
        self.prev_frame = None
        self.prev_bboxes = []  # (cls, conf, ymin, xmin, ymax, xmax) in [0,1] range
        self.internal_tracker = cv.MultiTracker_create()
        self.original_info = []
        self.tracker_name = name

    def reset_self(self):

        self.prev_frame = None
        self.prev_bboxes = None # (cls, conf, ymin, xmin, ymax, xmax) in [0,1] range
        self.internal_tracker = None
        self.original_info = None
        self.__init__(ds=self.ds, name=self.tracker_name)

    def createTrackerByName(self, trackerType):

        # Create a tracker based on tracker name
        trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN',
                        'MOSSE', 'CSRT']
        trackerType = trackerType.upper()
        if trackerType == trackerTypes[0]:
            tracker = cv.TrackerBoosting_create()
        elif trackerType == trackerTypes[1]:
            tracker = cv.TrackerMIL_create()
        elif trackerType == trackerTypes[2]:
            tracker = cv.TrackerKCF_create()
        elif trackerType == trackerTypes[3]:
            tracker = cv.TrackerTLD_create()
        elif trackerType == trackerTypes[4]:
            tracker = cv.TrackerMedianFlow_create()
        elif trackerType == trackerTypes[5]:
            tracker = cv.TrackerGOTURN_create()
        elif trackerType == trackerTypes[6]:
            tracker = cv.TrackerMOSSE_create()
        elif trackerType == trackerTypes[7]:
            tracker = cv.TrackerCSRT_create()
        else:
            tracker = None
            print('Incorrect tracker name')
            print('Available trackers are:')
            for t in trackerTypes:
                print(t)
        return tracker

    def resize(self,input_img):

        if self.ds > 1:
            height, width,_ = input_img.shape
            prvs_img = cv.resize(input_img, (int(width / self.ds),
                                             int(height / self.ds)))
            return prvs_img
        else:
            return input_img

    # transfer from (ymin,xmin,ymax,xmax) in [0,1]
    #            to (xmin,ymin,width,height) in original size
    def change_to_tracker_format(self,box,frame):

        height, width, _ = frame.shape
        new_input_boxes = (int(np.round(box[3] * width)), 
                           int(np.round(box[2] * height)), 
                           int(np.round((box[5] - box[3]) * width)),
                           int(np.round((box[4] - box[2]) * height)))
        return new_input_boxes

    # transfer from (xmin,ymin,width,height) in original size
    #            to (ymin,xmin,ymax,xmax) in [0,1]
    def recover_to_output_format(self,box,frame):

        height, width, _ = frame.shape
        new_output_boxes = (box[1] / height, box[0] / width,
                            (box[1] + box[3]) / height,
                            (box[0] + box[2]) / width)
        new_output_boxes = (max(new_output_boxes[0],0),
                            max(new_output_boxes[1],0),
                            min(new_output_boxes[2],1),
                            min(new_output_boxes[3],1))
        return new_output_boxes

    def set_prev_frame(self, frame = None, bboxes = []):

        if self.prev_frame is not None:
            self.prev_frame = self.resize(frame)
        else:
            # do initial tracking
            self.prev_frame = self.resize(frame)
            for box in bboxes:
                new_input_boxes = self.change_to_tracker_format(box,self.prev_frame)
                if self.tracker_name == 'csrt' and \
                   new_input_boxes[2] * new_input_boxes[3] <= 10:
                    continue
                self.internal_tracker.add(self.createTrackerByName(self.tracker_name),
                                          self.prev_frame,new_input_boxes)
                self.original_info.append((box[0],box[1]))

    def track(self, curr_frame):

        curr_frame = self.resize(curr_frame)
        new_boxes = []
        success, boxes = self.internal_tracker.update(curr_frame)
        for origin_info, box in zip(self.original_info, boxes):
            if success:
                new_out_box = self.recover_to_output_format(box,curr_frame)
                final_box = origin_info + new_out_box
                new_boxes.append(final_box)
        return new_boxes

class FlowRawTracker:
    def __init__(self, ds = 1, anchor = "fixed", mode = "bbox_median"):
        self.ds = ds
        self.prev_frame = None
        self.prev_bboxes = [] # (cls, conf, ymin, xmin, ymax, xmax) in [0,1]
        self.anchor = anchor
        self.mode = mode

    def set_prev_frame(self, frame = None, bboxes = []):
        if not frame is None:
            prvs_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            if self.ds > 1:
                height, width = prvs_img.shape
                prvs_img = cv.resize(prvs_img, (int(width/self.ds),
                                                int(height/self.ds)))
            self.prev_frame = prvs_img
        self.prev_bboxes = bboxes

    def track(self, curr_frame):
        next_img = cv.cvtColor(curr_frame, cv.COLOR_BGR2GRAY)
        height, width = next_img.shape
        if self.ds > 1:
            next_img = cv.resize(next_img, (int(width/self.ds), int(height/self.ds)))
        height, width = next_img.shape
        
        img_flow = cv.calcOpticalFlowFarneback(prev = self.prev_frame, next = next_img,
                                       flow = None, pyr_scale = 0.5,
                                       levels = 3, winsize = 15, iterations = 3,
                                       poly_n = 5, poly_sigma = 1.1,
                                       flags = cv.OPTFLOW_FARNEBACK_GAUSSIAN)

        # Extract flow output
        new_bboxes = []
        for cls, conf, ymin, xmin, ymax, xmax in self.prev_bboxes:
            sig_pts0, sig_pts = [], []

            # First, find out the "significant" flows
            for x in range(int(xmin*width), int(xmax*width), 4):
                for y in range(int(ymin*height), int(ymax*height), 4):
                    dx, dy = int(img_flow[y, x, 0]), int(img_flow[y, x, 1])
                    st_pt = (x, y)        # width(y-dim) first
                    en_pt = (x+dx, y+dy)  # width(y-dim) first
                    if (abs(dx)+abs(dy) >= 5/self.ds):
                        sig_pts0.append((x, y, x+dx, y+dy))

            # Reject anapolies flows
            abs_xs = [abs(sig_pt[2] - sig_pt[0]) for sig_pt in sig_pts0]
            abs_ys = [abs(sig_pt[3] - sig_pt[1]) for sig_pt in sig_pts0]
            if len(abs_xs) >= 1 and len(abs_ys) >= 1:
                mean_x, std_x = np.mean(abs_xs), np.std(abs_xs)
                mean_y, std_y = np.mean(abs_ys), np.std(abs_ys)
                
                for x, y, x_pl_dx, y_pl_dy in sig_pts0:
                    if abs(x_pl_dx - x) - mean_x <= 2 * std_x and \
                       abs(y_pl_dy - y) - mean_y <= 2 * std_y:
                        sig_pts.append((x, y, x_pl_dx, y_pl_dy))
            else:
                sig_pts = sig_pts0

            # If the box moves
            if len(sig_pts) > 0:
                if self.mode == "pixel":
                    # new box is based on per-pixel mins, maxs
                    sig_xs = [sig_pt[2]/width for sig_pt in sig_pts]
                    sig_ys = [sig_pt[3]/height for sig_pt in sig_pts]
                    _ymin, _xmin = min(sig_ys), min(sig_xs), 
                    _ymax, _xmax = max(sig_ys), max(sig_xs)
                elif self.mode == "bbox":
                    # new box is based on mean dx, dy
                    dxs = [sig_pt[2] - sig_pt[0] for sig_pt in sig_pts]
                    dys = [sig_pt[3] - sig_pt[1] for sig_pt in sig_pts]
                    mean_x, mean_y = np.mean(dxs)/width, np.mean(dys)/height
                    _ymin, _xmin = ymin + mean_y, xmin + mean_x
                    _ymax, _xmax = ymax + mean_y, xmax  + mean_x
                elif self.mode == "bbox_median":
                    # new box is based on median dx, dy
                    dxs = [sig_pt[2] - sig_pt[0] for sig_pt in sig_pts]
                    dys = [sig_pt[3] - sig_pt[1] for sig_pt in sig_pts]
                    median_x, median_y = np.median(dxs)/width, np.median(dys)/height
                    _ymin, _xmin = ymin + median_y, xmin + median_x
                    _ymax, _xmax = ymax + median_y, xmax  + median_x
                else:
                    print("Error in mode of the object tracker.")
                    return

                # Make sure the new box is trancated within the frame
                #  -- [0, width-1] x [0, height-1]
                _xmin, _ymin = max(0, _xmin), max(0, _ymin)
                _xmax, _ymax  = min(1, _xmax), min(1, _ymax)
                if _ymin < _ymax and _xmin < _xmax:  
                    new_bboxes.append((cls, conf, _ymin, _xmin, _ymax, _xmax))
            else:
                new_bboxes.append((cls, conf, ymin, xmin, ymax, xmax))

        if self.anchor != "fixed": # then the reference frame is "moving"
            self.prev_bboxes = new_bboxes
            self.prev_frame = next_img
        return new_bboxes

