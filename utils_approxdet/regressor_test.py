# Test the prediction modules.
# Usage: python3 utils_approxdet/regressor_test.py

from utils_approxdet.regressor import AccuracyPredictor, AccuracyPredictorVideowise
from utils_approxdet.regressor import TrackerLatencyPredictor, DNNLatencyPredictor
from utils_approxdet.regressor import SwitchingLatencyPredictor
import time

def TrackerLatencyPredictorTest(model_file):

    print("TrackerLatencyPredictorTest() begins.")
    tr_lat_pred = TrackerLatencyPredictor(model_file)

    for tracker, ds in [("medianflow", 4), ("medianflow", 2), ("medianflow", 1),
                        ("kcf", 4), ("csrt", 4), ("bboxmedianfixed", 4)]:
        tr_init_lat = tr_lat_pred.predict_init(num_obj=1, avg_size=360.95,
                                               width=1280, height=720,
                                               core=0, mb=0, gpu=0,
                                               tracker=tracker, ds=ds)
        print("obj_size, tracker, ds=(360.95, {}, {}), ".format(tracker, ds) + \
              "L_init = {:.2f} msec".format(tr_init_lat))
        tr_init_lat = tr_lat_pred.predict_init(num_obj=1, avg_size=22.56,
                                               width=1280, height=720,
                                               core=0, mb=0, gpu=0,
                                               tracker=tracker, ds=ds)
        print("obj_size, tracker, ds=(22.56, {}, {}), ".format(tracker, ds) + \
              "L_init = {:.2f} msec".format(tr_init_lat))

        tr_tr_lat = tr_lat_pred.predict_tr(num_obj=1, avg_size=360.95,
                                           width=1280, height=720,
                                           core=0, mb=0, gpu=0,
                                           tracker=tracker, ds=ds)
        print("obj_size, tracker, ds=(360.95, {}, {}), ".format(tracker, ds) + \
              "L_tr = {:.2f} msec".format(tr_tr_lat))
        tr_tr_lat = tr_lat_pred.predict_tr(num_obj=1, avg_size=22.56,
                                           width=1280, height=720,
                                           core=0, mb=0, gpu=0,
                                           tracker=tracker, ds=ds)
        print("obj_size, tracker, ds=(22.56, {}, {}), ".format(tracker, ds) + \
              "L_tr = {:.2f} msec".format(tr_tr_lat))

    for tracker, ds in [("medianflow", 4), ("medianflow", 2), ("medianflow", 1),
                        ("kcf", 4), ("csrt", 4), ("bboxmedianfixed", 4)]:
        tr_init_lat = tr_lat_pred.predict_init(num_obj=0, avg_size=0,
                                               width=480, height=360,
                                               core=0, mb=0, gpu=0,
                                               tracker=tracker, ds=ds)
        print("num_obj, tracker, ds=(0, {}, {}), ".format(tracker, ds) + \
              "L_init = {:.2f} msec".format(tr_init_lat))

        tr_tr_lat = tr_lat_pred.predict_tr(num_obj=0, avg_size=0,
                                           width=480, height=360,
                                           core=0, mb=0, gpu=0,
                                           tracker=tracker, ds=ds)
        print("num_obj, tracker, ds=(0, {}, {}), ".format(tracker, ds) + \
              "L_tr = {:.2f} msec".format(tr_tr_lat))

    tr_tr_lat = tr_lat_pred.predict_tr(num_obj=0, avg_size=0,
                                       width=1280, height=720,
                                       core=6, mb=3600, gpu=0,
                                       tracker='medianflow', ds=4)
    print("tracking latency = {:.2f} msec".format(tr_tr_lat))

    tr_tr_lat = tr_lat_pred.predict_tr(num_obj=1, avg_size=200,
                                       width=1280, height=720,
                                       core=6, mb=3600, gpu=50,
                                       tracker='medianflow', ds=4)
    print("tracking latency = {:.2f} msec".format(tr_tr_lat))

    tr_tr_lat = tr_lat_pred.predict_tr(num_obj=0, avg_size=0,
                                       width=1280, height=720,
                                       core=6, mb=3600, gpu=50,
                                       tracker='medianflow', ds=4)
    print("tracking latency = {:.2f} msec".format(tr_tr_lat))
    print("TrackerLatencyPredictorTest() ends.\n")

def TrackerLatencyPredictorOverhead(model_file):

    print("TrackerLatencyPredictorOverhead() begins.\n  Overhead:")
    tr_lat_pred = TrackerLatencyPredictor(model_file)
    for i in range(10):
        time_st = time.time()
        _, _ = tr_lat_pred.predict(num_obj=2, avg_size=220,
                                   width=1280, height=720,
                                   core=i, mb=0, gpu=0)
        print("  {:.2f} msec".format((time.time()-time_st)*1e3))
    print("TrackerLatencyPredictorOverhead() ends.\n")

def DNNLatencyPredictorTest(model_file):

    print("DNNLatencyPredictorTest() begins.")
    det_lat_pred = DNNLatencyPredictor(model_file)
    time_st = time.time()
    lat = det_lat_pred.predict(height=720, width=1280, cpu_contention=0,
                               gpu_contention=0)
    overhead = (time.time()-time_st)*1e3
    print("len(lat) = {}, overhead = {:.2f} ms.".format(len(lat), overhead))
    print("DNNLatencyPredictorTest() ends.\n")

def AccuracyPredictorTest(model_file):

    print("AccuracyPredictorTest() begins.")
    acc_pred = AccuracyPredictor(model_file = model_file)

    # predict one-time
    si, shape, nprop, tracker, ds = 20, 448, 20, "medianflow", 4
    accuracy = acc_pred.predict_onemodel(si, shape, nprop, tracker, ds)
    print("accuracy = {}".format(accuracy))

    # predict for all
    overheads = []
    for i in range(10):
        time_st = time.time()
        per_branch_accuracy = acc_pred.predict()
        overheads.append((time.time()-time_st)*1e3)
    print("len(per_branch_accuracy) = {}".format(len(per_branch_accuracy)))
    print("overheads in ms: {}".format(overheads))
    print("AccuracyPredictorTest() ends.\n")

def AccuracyPredictorVideoWiseTest(model_file):

    print("AccuracyPredictorVideoWiseTest() begins.")
    acc_pred = AccuracyPredictorVideowise(model_file)

    # predict one-time
    si, shape, nprop, tracker, ds = 20, 448, 20, "medianflow", 4
    movement = 0.0001
    accuracy = acc_pred.predict_onemodel(si, shape, nprop, tracker, ds, movement)
    print("accuracy = {}".format(accuracy))

    # predict for all
    overheads = []
    for i in range(10):
        time_st = time.time()
        per_branch_accuracy = acc_pred.predict(movement)
        overheads.append((time.time() - time_st) * 1e3)
    print("len(per_branch_accuracy) = {}".format(len(per_branch_accuracy)))
    print("overheads in ms: {}".format(overheads))
    print("AccuracyPredictorVideoWiseTest() ends.\n")

def SwitchingLatencyPredictorTest(model_file):

    print("SwitchingLatencyPredictorTest() begins.")
    oh_pred = SwitchingLatencyPredictor(model_file)

    # predict one-time: (si, shape, nprop, tracker, ds)
    current_config = 1, 576, 100, "medianflow", 4
    overhead = oh_pred.predict(current_config)
    print("  switching overhead = {}".format(overhead))
    print("SwitchingLatencyPredictorTest() ends.\n")

if __name__ == "__main__":

    TrackerLatencyPredictorTest('models/ApproxDet_LatTr.pb')
    TrackerLatencyPredictorOverhead('models/ApproxDet_LatTr.pb')
    DNNLatencyPredictorTest('models/ApproxDet_LatDet.pb')
    AccuracyPredictorTest('models/ApproxDet_Acc.pb')
    AccuracyPredictorVideoWiseTest('models/ApproxDet_Acc2.pb')
    SwitchingLatencyPredictorTest('models/ApproxDet_LatSw.pb')

