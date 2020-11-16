# Given a model and input features, output latency/accuracy predictions.

import numpy as np
import sklearn.linear_model
import sklearn.tree
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
import pickle

class TrackerLatencyPredictor:

    def __init__(self, model_file):

        self.list_tracker = [("medianflow", 4), ("medianflow", 2),
                             ("medianflow", 1), ("kcf", 4), 
                             ("csrt", 4), ("bboxmedianfixed", 4)]
        self.transform = PolynomialFeatures(2)
        self.init_models, self.tr_models = {}, {}
        self.init_coeff, self.init_intercept = [], []
        self.tracking_coeff, self.tracking_intercept = [], []

        all_models = pickle.load(open(model_file,'rb'))
        for tracker, ds in self.list_tracker:
            key = "{}_ds{}_init".format(tracker, ds)
            if key in all_models:
                self.init_models[(tracker, ds)] = all_models[key]
                self.init_coeff.append(all_models[key].coef_)
                self.init_intercept.append(all_models[key].intercept_)
            else:
                print("Error in loading latency prediction model.")
                return

            key = "{}_ds{}_tracking".format(tracker, ds)
            if key in all_models:
                self.tr_models[(tracker, ds)] = all_models[key]
                self.tracking_coeff.append(all_models[key].coef_)
                self.tracking_intercept.append(all_models[key].intercept_)
            else:
                print("Error in loading latency prediction model.")
                return

        self.init_coeff = np.array(self.init_coeff)
        self.init_intercept = np.array(self.init_intercept)
        self.tracking_coeff = np.array(self.tracking_coeff)
        self.tracking_intercept = np.array(self.tracking_intercept)

    # one-time per-model prediction
    def predict_init(self, num_obj, avg_size, width, height, 
                     core=0, mb=0, gpu=0, tracker="medianflow", ds=4):

        if (tracker, ds) in self.init_models:
            feature = [num_obj, avg_size, width, height, core, mb, gpu]
            X = np.array(feature).reshape(1, -1)  # (1, 7) shape
            X = self.transform.fit_transform(X)
            return self.init_models[(tracker, ds)].predict(X)[0].tolist()
        else:
            print("Cannot find latency prediction model.")

    # one-time per-model prediction
    def predict_tr(self, num_obj, avg_size, width, height, 
                   core=0, mb=0, gpu=0, tracker = "medianflow", ds = 4):

        if (tracker, ds) in self.tr_models:
            feature = [num_obj, avg_size, width, height, core, mb, gpu]
            X = np.array(feature).reshape(1, -1)  # (1, 7) shape
            X = self.transform.fit_transform(X)
            return self.tr_models[(tracker, ds)].predict(X)[0].tolist()
        else:
            print("Cannot find latency prediction model.")

    # faster implementation
    def batch_prediction(self, num_obj, avg_size, width, height,
                         core=0, mb=0, gpu=0):

        feature = [num_obj, avg_size, width, height, core, mb, gpu]
        X = np.array(feature).reshape(1, -1)  # (1, 7) shape
        X = self.transform.fit_transform(X).squeeze()
        init_time = np.dot(self.init_coeff, X) + self.init_intercept  # (6, 1)
        tr_time = np.dot(self.tracking_coeff, X) + self.tracking_intercept
        return init_time.tolist(), tr_time.tolist()

    def predict(self, num_obj, avg_size, width, height, core=0, mb=0, gpu=0):

        latency_init = [self.predict_init(num_obj, avg_size, width, height,
                                          core, mb, gpu, tracker, ds) \
                        for (tracker, ds) in self.list_tracker]
        latency_tr = [self.predict_tr(num_obj, avg_size, width, height,
                                      core, mb, gpu, tracker, ds) \
                      for (tracker, ds) in self.list_tracker]
        # latency_init, latency_tr = self.batch_prediction(num_obj, avg_size,
        #                                                  width, height,
        #                                                  core, mb, gpu)
        return latency_init, latency_tr

class DNNLatencyPredictor:

    def __init__(self, model_file, sis = [1, 2, 4, 8, 20, 50, 100],
                 shapes = [224, 320, 448, 576],
                 nprops = [1, 3, 5, 10, 20, 50, 100]):

        model = pickle.load(open(model_file, "rb"))
        self.coeff_array = [model[(n, s)][0] for s in shapes for n in nprops]
        self.bias_array = [model[(n, s)][1] for s in shapes for n in nprops]
        self.coeff_array = np.array(self.coeff_array) # (28, 15)
        self.bias_array = np.array(self.bias_array)   # (28,)
        self.transform = PolynomialFeatures(2)        # (4,) -> (15,)
        self.sis = sis

    def predict(self, height, width, cpu_contention=0, gpu_contention=0):

        features = [height, width, cpu_contention, gpu_contention] # (4,)
        features = np.array(features).reshape(1, -1)
        features = self.transform.fit_transform(features).squeeze() # (15,)
        now_time = np.dot(self.coeff_array, features) + self.bias_array
        now_time = now_time.reshape(4, 7) # (shape, nprop), arranged (4, 7) shape
        #(si, shape, nprop) arranged , (7, 4, 7) shape
        with_ds_array = np.stack([now_time/self.sis[i] for i in range(7)], axis=0)
        #(si, shape, nprop, tracker) arranged, (7, 4, 7, 6) shape
        final_array = np.repeat(with_ds_array[:, :, :, np.newaxis], 6, axis=3)
        return list(final_array.flatten())

class AccuracyPredictor:

    def __init__(self, model_file):

        self.model = pickle.load(open(model_file, "rb"))
        tracker_list = ["none", "medianflow", "kcf", "csrt", "bboxmedianfixed"]
        tracker_array = np.array(tracker_list).reshape(-1, 1)
        onehot_trackerencoder = OneHotEncoder(sparse=False)
        one_hot_encoded = onehot_trackerencoder.fit_transform(tracker_array)
        self.tracker_dict = {}
        for i, tracker in enumerate(tracker_list):
            self.tracker_dict[tracker] = list(one_hot_encoded[i, :])

        configs = [(si, s, n, t, ds) for si in [1, 2, 4, 8, 20, 50, 100] \
                   for s in [224, 320, 448, 576] \
                   for n in [1, 3, 5, 10, 20, 50, 100] \
                   for t, ds in [("medianflow", 4), ("medianflow", 2),
                                 ("medianflow", 1), ("kcf", 4),
                                 ("csrt", 4), ("bboxmedianfixed", 4)]]
        self.config_X = [[nprop, shape, si, *self.tracker_dict[tracker], ds] \
                         for (si, shape, nprop, tracker, ds) in configs]
        self.config_X = np.array(self.config_X)
        self.acc = (self.model.predict(self.config_X)/100).tolist()

    def predict_onemodel(self, si, shape, nprop, tracker, ds):

        feature = [nprop, shape, si, *self.tracker_dict[tracker], ds]
        X = np.array(feature).reshape(1, -1)
        return self.model.predict(X)[0]/100

    def predict(self): # norm. accuracy compared to the base branch

        return self.acc

class AccuracyPredictorVideowise:

    def __init__(self, model_file):

        self.model = pickle.load(open(model_file, "rb"))
        self.config_index, self.coeff, self.intercept = {}, [], []
        for i, config in enumerate(self.model.keys()):
            self.config_index[config] = i
            self.coeff.append(self.model[config][-2][0])
            self.intercept.append(self.model[config][-1])
        self.coeff = np.array(self.coeff)
        self.intercept = np.array(self.intercept)

    def predict_onemodel(self, si, shape, nprop, tracker, ds, movement):

        config = (nprop, shape, si, tracker, ds)
        now_coeff = self.coeff[self.config_index[config]]
        now_intercept = self.intercept[self.config_index[config]]
        return (now_coeff * movement + now_intercept) / 100

    def predict(self, movement): #return all accuracies according to the configs

        # configs = [(si, s, n, t, ds) for si in [1, 2, 4, 8, 20, 50, 100] \
        #            for s in [224, 320, 448, 576] \
        #            for n in [1, 3, 5, 10, 20, 50, 100] \
        #            for t, ds in [("medianflow", 4), ("medianflow", 2),
        #                          ("medianflow", 1), ("kcf", 4),
        #                          ("csrt", 4), ("bboxmedianfixed", 4)]]
        now_movement = np.ones_like(self.coeff) * movement # (1176, 1)
        ans = (self.coeff * now_movement + self.intercept) / 100
        return ans.tolist() # (1176, 1)

class SwitchingLatencyPredictor:

    def __init__(self, model_file):

        self.list_sis = [1, 2, 4, 8, 20, 50, 100]
        self.list_shapes = [224, 320, 448, 576]
        self.list_nprops = [1, 3, 5, 10, 20, 50, 100]
        self.list_trackers = [("medianflow", 4), ("medianflow", 2), ("medianflow", 1),
                             ("kcf", 4), ("csrt", 4), ("bboxmedianfixed", 4)]

        # 28x28 np 2D array ordered in "detection_branches"
        self.table = pickle.load(open(model_file, "rb"))
        self.detection_branches = [(n, s) for n in [1, 3, 5, 10, 20, 50, 100] \
                                          for s in [224, 320, 448, 576]]

    def predict(self, current_config):

        if not current_config[0]:
            return [0 for _ in range(1176)]
        # Predict the switching latency from current_config to any configs
        _, current_shape, current_nprop, _, _ = current_config
        from_branch_id = self.detection_branches.index((current_nprop, current_shape))

        # si, shape, nprop, tracker_ds
        switching_overhead = np.zeros((7, 4, 7, 6)) 
        for idx1, si in enumerate(self.list_sis):
            for idx2, shape in enumerate(self.list_shapes):
                for idx3, nprop in enumerate(self.list_nprops):
                    for idx4, (tracker, ds) in enumerate(self.list_trackers):
                        to_branch_id = self.detection_branches.index((nprop, shape))
                        switching_overhead[idx1, idx2, idx3, idx4] = \
                          self.table[from_branch_id, to_branch_id]
        return list(switching_overhead.flatten())

