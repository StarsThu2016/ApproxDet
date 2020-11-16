# Use nearest logs to estimate the contention level
# Usage: python3 utils_approxdet/contention_sensor.py

import os, pickle
import numpy as np
from collections import defaultdict

class ContentionSensor:

    def __init__(self, model_file):

        d = pickle.load(open(model_file, 'rb'))
        self.hw_list = d["hw_list"]
        self.detection_conf_list = d["detection_conf_list"]
        self.tracker_conf_list = d["tracker_conf_list"]
        self.detection_latency_dict = d["detection_latency_dict"]
        self.tracker_init_latency_dict = d["tracker_init_latency_dict"]
        self.tracker_tracking_latency_dict = d["tracker_tracking_latency_dict"]
        self.last_level = {"cpu_level": 0, "mem_bw_level": 0, "gpu_level": 0}

    def sense(self, si, shape, nprop, tracker_type, height, width, latency_list,
              operation_list):

        if len(operation_list) == 0:
            return self.last_level

        # cast (height, width) to one that exists in our log
        if (height, width) not in self.hw_list:
            distance = [abs(h*w-height*width) for h, w in self.hw_list]
            height, width = self.hw_list[np.argmin(distance)]

        # estimate the gpu contention level
        mean_detection_time = np.mean([lat for (lat, op) in \
          zip(latency_list, operation_list) if op == 'detection'])
        detection_log = self.detection_latency_dict[(height, width, nprop, shape)]
        distance = [abs(detection_log[config] - mean_detection_time) \
                    for config in self.detection_conf_list]
        now_gpu_config = self.detection_conf_list[np.argmin(distance)]

        # estimate the cpu/mem_bw contention level
        lat_tr = np.mean([lat for (lat, op) in \
          zip(latency_list, operation_list) if op == 'tracking'])
        trinit_log = self.tracker_init_latency_dict[(height, width, tracker_type)]
        trtr_log = self.tracker_tracking_latency_dict[(height, width, tracker_type)]
        n_det = operation_list.count("detection")
        n_tr = operation_list.count("tracking")
        distance = [abs(trinit_log[config]*n_det/n_tr+trtr_log[config]-lat_tr) \
                    for config in self.tracker_conf_list]
        now_cpu_config = self.tracker_conf_list[np.argmin(distance)]

        self.last_level = {"cpu_level": now_cpu_config[0], 
                           "mem_bw_level": now_cpu_config[1], 
                           "gpu_level": now_gpu_config}
        return self.last_level

if __name__ == '__main__':

    b = ContentionSensor('models/ApproxDet_CS.pb')
    print(b.sense(nprop=100, shape=576, tracker_type='medianflow_ds4',
          height=480, width=872, si=1, latency_list=[600,100],
          operation_list=['detection','tracking']))
    print(b.sense(nprop=100, shape=576, tracker_type='medianflow_ds4',
          height=720, width=1280, si=1, latency_list=[800,100],
          operation_list=['detection','tracking']))
    print(b.sense(nprop=100, shape=576, tracker_type='medianflow_ds4',
          height=720, width=1280, si=1, latency_list=[1000,100],
          operation_list=['detection','tracking']))
    print(b.sense(nprop=100, shape=576, tracker_type='medianflow_ds4',
          height=720, width=1280, si=1, latency_list=[1500,100],
          operation_list=['detection','tracking']))

