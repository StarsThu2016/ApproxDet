# The scheduler module of the online ApproxDet

from utils_approxdet.regressor import AccuracyPredictor, AccuracyPredictorVideowise
from utils_approxdet.regressor import TrackerLatencyPredictor, DNNLatencyPredictor
from utils_approxdet.regressor import SwitchingLatencyPredictor
from utils_approxdet.contention_sensor import ContentionSensor
import numpy as np

class Scheduler:

    def __init__(self, user_requirement = {"latency": 200}):

        # content features for tracker latency model
        self.last_nobj = 0             # this always gets updated
        self.last_avgsize = 0.         # this always gets updated
        self.last_nobj_ondet = 0       # this gets updated on the detection frame
        self.last_avgsize_ondet = 0.   # this gets updated on the detection frame
        self.feature_nobj = 0
        self.feature_avgsize = 0.

        # content features for content-agonostic accuracy model
        self.past_movements = []

        # logging for contention sensor
        #  flush policy: config change // new_video's first update
        #  re-write policy: hit the retention value
        self.latency_logging = {"detection": [], "tracking": []}
        self.latency_logging_hw = (None, None)
        self.latency_logging_retention = {"detection": 1, "tracking": 10}

        # hard-coded parameters
        self.list_sis = [1, 2, 4, 8, 20, 50, 100]
        self.list_shapes = [224, 320, 448, 576]
        self.list_nprops = [1, 3, 5, 10, 20, 50, 100]
        self.list_trackers = [("medianflow", 4), ("medianflow", 2), ("medianflow", 1),
                             ("kcf", 4), ("csrt", 4), ("bboxmedianfixed", 4)]
        self.config_list = [(si, shape, nprop, tracker, ds) \
                            for si in self.list_sis \
                            for shape in self.list_shapes \
                            for nprop in self.list_nprops \
                            for tracker, ds in self.list_trackers]
        self.scheduler_internal = 10

        # operation control to make sure the logic is correct
        self.last_video = None          # the name of last video
        self.last_det_frame_idx = None  # the index of last detection frame
        self.last_sch_frame_idx = None  # the index of last scheduler frame
        self.config = (None, None, None, None, None) # (si, shape, nprop, tracker, ds)
        self.selected_accuracy, self.selected_latency = 0, 0

        # sub-modules
        self.accuracy_predictor = \
          AccuracyPredictor(model_file="models/ApproxDet_Acc.pb")
        self.accuracy_predictor_videowise = \
          AccuracyPredictorVideowise(model_file="models/ApproxDet_Acc2.pb")
        self.DNN_latency_predictor = \
          DNNLatencyPredictor(model_file="models/ApproxDet_LatDet.pb")
        self.tracker_latency_predictor = \
          TrackerLatencyPredictor(model_file="models/ApproxDet_LatTr.pb")
        self.switching_latency_predictor = \
          SwitchingLatencyPredictor(model_file="models/ApproxDet_LatSw.pb")
        self.contention_sensor = \
          ContentionSensor(model_file="models/ApproxDet_CS.pb")

        # user requirement target
        self.user_requirement = user_requirement

    def is_det_frame(self, vid_name, frame_idx):

        # return is_det_frame, is_scheduler_frame, is_new_video
        if vid_name != self.last_video or frame_idx < self.last_det_frame_idx \
          or frame_idx < self.last_sch_frame_idx:
            self.last_video = vid_name
            self.past_movements = []
            self.last_det_frame_idx = frame_idx
            self.last_sch_frame_idx = frame_idx
            return True, True, True
        elif (frame_idx-self.last_sch_frame_idx)%self.scheduler_internal == 0:
            self.last_det_frame_idx = frame_idx
            self.last_sch_frame_idx = frame_idx
            return True, True, False
        elif (frame_idx-self.last_det_frame_idx)%self.config[0] == 0:
            self.last_det_frame_idx = frame_idx
            return True, False, False
        else:
            return False, False, False

    def schedule(self, contention_levels, height, width):

        # Accuracy model: returns a list of 1176 length
        if len(self.past_movements) >= 300:
            movement = np.mean(self.past_movements)
            per_branch_accuracy = self.accuracy_predictor_videowise.predict(movement)
        else:
            per_branch_accuracy = self.accuracy_predictor.predict()

        # Detection latency model: returns a list of 1176 length
        per_branch_DNN_latency = self.DNN_latency_predictor.predict(height=height,
          width=width, gpu_contention=contention_levels["gpu_level"])

        # Tracker latency model: returns two lists of 6 length, then resize to 1176
        self.feature_nobj, self.feature_avgsize = \
          (self.last_nobj, self.last_avgsize) if self.last_nobj > 0 \
          else (self.last_nobj_ondet, self.last_avgsize_ondet)
        latency_init, latency_tr = self.tracker_latency_predictor.predict(\
          self.feature_nobj, self.feature_avgsize, width, height,
          core = contention_levels["cpu_level"], 
          mb = contention_levels["mem_bw_level"],
          gpu = contention_levels["gpu_level"])
        per_branch_latency_init = np.tile(latency_init, (7,4,7,1))
        per_branch_latency_tr = np.tile(latency_tr, (7,4,7,1))
        for idx, si in enumerate(self.list_sis):
            per_branch_latency_tr[idx, :, :, :] *= ((si-1)/si)
            per_branch_latency_init[idx, :, :, :] *= (1./si)
        per_branch_latency_init[:, :, :, 4] = float("inf")
        per_branch_latency_tr[:, :, :, 4] = float("inf")
        per_branch_latency_init = list(per_branch_latency_init.flatten())
        per_branch_latency_tr = list(per_branch_latency_tr.flatten())

        # Switching latency model: returns a list of 1176 length
        per_branch_switching_latency = self.switching_latency_predictor.predict(self.config)
        stay_window = [100 for _ in range(1176)]

        # Pareto Frountier Searching
        assert len(self.config_list) == len(per_branch_accuracy)
        assert len(self.config_list) == len(per_branch_switching_latency)
        assert len(self.config_list) == len(per_branch_latency_init)
        assert len(self.config_list) == len(per_branch_latency_tr)
        assert len(self.config_list) == len(per_branch_DNN_latency)
        acc_lat_config_tups = [(acc, lat_DNN+lat_init+lat_tr+lat_sw/sw, config) \
          for config, acc, lat_sw, sw, lat_init, lat_tr, lat_DNN in \
          zip(self.config_list, per_branch_accuracy, per_branch_switching_latency,
              stay_window, per_branch_latency_init, per_branch_latency_tr,
              per_branch_DNN_latency)]

        # ascending order for latency
        acc_lat_config_tups = sorted(acc_lat_config_tups, key = lambda x: x[1])
        acc_lat_config_frontiers, best_acc = [], 0
        for acc, lat, config in acc_lat_config_tups:
            if acc > best_acc:
                best_acc = acc
                acc_lat_config_frontiers.append((acc, lat, config))
                print(acc, lat, config)
        acc_lat_config_satisfied = [(acc, lat, config) for acc, lat, config in \
          acc_lat_config_frontiers if lat <= self.user_requirement["latency"]/1.3]
        if acc_lat_config_satisfied:  # Found one to satisfy
            self.selected_accuracy, self.selected_latency, new_config = \
              acc_lat_config_satisfied[-1] # the most accuracte one
        else:
            self.selected_accuracy, self.selected_latency, new_config = \
              acc_lat_config_frontiers[0]  # the fastest one

        # Trick to handle "multiple same config if si == 1":
        #   replace the tracker and ds in new_config
        if new_config[0] == 1:  # si == 1
            si, shape, nprop, _, _ = new_config
            _, _, _, old_tracker, old_ds = self.config
            new_config = (si, shape, nprop, old_tracker, old_ds) if old_tracker \
              else (si, shape, nprop, "medianflow", 4)

        # Flush the latency logging, if a new config is twiggered
        if self.config != new_config:
            self.latency_logging = {"detection": [], "tracking": []}
            self.latency_logging_config = None

        # Endding, config = (si, shape, nprop, tracker, ds)
        if_switch_tracker = (self.config[4] != new_config[4])
        self.config = new_config

        # re-calibrate scheduler_internal with si
        self.scheduler_internal = max(self.config[0], 8) # min at 8
        return self.config, if_switch_tracker, self.selected_accuracy, \
          self.selected_latency, self.feature_nobj, self.feature_avgsize

    def get_last_config(self): # For tracking frame, return the last config

        return self.config, False, self.selected_accuracy, \
          self.selected_latency, self.feature_nobj, self.feature_avgsize

    def update(self, update_dict):

        if "user_requirement" in update_dict:
            self.user_requirement = update_dict["user_requirement"]
        if "last_avgsize" in update_dict:
            self.last_avgsize = update_dict["last_avgsize"]
        if "last_nobj" in update_dict:
            self.last_nobj = update_dict["last_nobj"]
        if "last_nobj_ondet" in update_dict:
            self.last_nobj_ondet = update_dict["last_nobj_ondet"]
        if "last_avgsize_ondet" in update_dict:
            self.last_avgsize_ondet = update_dict["last_avgsize_ondet"]
        if "past_movement" in update_dict:
            self.past_movements.append(update_dict["past_movement"])
        if "if_new_video" in update_dict:
            # For a new video, flush the log since height/width changed
            self.latency_logging = {"detection": [], "tracking": []}
            self.latency_logging_config = None
            self.latency_logging_hw = update_dict["height"], update_dict["width"]
        if "latency_log_type" in update_dict:
            update_type = update_dict["latency_log_type"]
            update_value = update_dict["latency_log_value"]
            self.latency_logging[update_type].append(update_value)
            if len(self.latency_logging[update_type]) > \
              self.latency_logging_retention[update_type]:
                self.latency_logging[update_type].pop(0)

    def sense_contention(self):

        # self.config = (si, shape, nprop, tracker, ds)
        nprop, shape, si = self.config[2], self.config[1], self.config[0]
        tracker_type = "{}_ds{}".format(self.config[3], self.config[4])
        height, width = self.latency_logging_hw
        latency_list = self.latency_logging["detection"] + self.latency_logging["tracking"]
        operation_list = ["detection"] * len(self.latency_logging["detection"]) + \
                         ["tracking"] * len(self.latency_logging["tracking"])
        return self.contention_sensor.sense(si, shape, nprop, tracker_type, 
          height, width, latency_list, operation_list)

