'''
ApproxDet: the online adaptive object detection framework for streaming videos.

Usage:
python3 ApproxDet_CG.py
python3 ApproxDet.py --input=test/VID_testimg_cs1.txt --preheat=1 \
  --dataset_prefix=/home/nvidia/ILSVRC2015/ \
  --weight=models/ApproxDet.pb \
  --output=test/VID_testset_ApproxDet_cs1.txt
'''

import argparse, os, time, tqdm
import numpy as np
from PIL import Image
import tensorflow as tf
from utils_approxdet.tracker import OpenCVTracker, FlowRawTracker
from utils_approxdet.detection_helper import load_graph_from_file
from utils_approxdet.detection_helper import output_dict_to_bboxes_single_img
from utils_approxdet.scheduler import Scheduler
from utils_approxdet.contention_generator_3d import contention_generator_launch
from utils_approxdet.contention_generator_3d import contention_generator_kill

def preheat(sess, dataset_prefix, tensor_frame, tensor_nprop, tensor_shape,
            output_tensor_dict):

    print("preheating")
    preheat_img_dir = os.path.join(dataset_prefix, 
      "Data/VID/train/ILSVRC2015_VID_train_0002/ILSVRC2015_train_00703000")
    shapes, nprops = [224, 320, 448, 576], [100, 50, 20, 10, 5, 3, 1]
    configs = [(shape, nprop) for shape in shapes for nprop in nprops \
               for run in range(3)]
    configs_w_path = [(*config, "{}/{:06d}.JPEG".format(preheat_img_dir, idx)) \
                      for idx, config in enumerate(configs)]
    for shape, nprop, full_path in tqdm.tqdm(configs_w_path):
        image_pil = Image.open(full_path)
        image_np = np.array(image_pil).astype(np.uint8)
        image_4D = np.expand_dims(image_np, axis=0)
        feed_dict = {tensor_frame: image_4D, tensor_nprop: nprop,
                     tensor_shape: shape}
        output_dict = sess.run(output_tensor_dict, feed_dict = feed_dict)

if __name__== "__main__":

    # Argument parsing
    parser = argparse.ArgumentParser(description='Evaluate ApproxDet.')
    parser.add_argument('--input', dest='input', required=True, 
      help='Meta-data of evaluated video frames.')
    parser.add_argument('--preheat', dest='preheat', 
      help='Whether we preheat the system.')
    parser.add_argument('--dataset_prefix', dest='dataset_prefix',
      help='The path to the dataset.')
    parser.add_argument('--weight', dest='weight', 
      help='The path to the weight file.')
    parser.add_argument('--output', dest='output', required=True,
      help='Output logging file.')
    args = parser.parse_args()

    # Load list of video frames
    with open(args.input) as f:
        lines = f.readlines()
    configs = [x.strip().split() for x in lines]
                
    # Output log files
    detoutput_filename = args.output.split(".")[0] + "_det.txt"
    latoutput_filename = args.output.split(".")[0] + "_lat.txt"
    fout_det = open(detoutput_filename, "w")
    fout_lat = open(latoutput_filename, "w")

    # Set TensorFlow config
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    # Load detection DNN
    detection_graph = load_graph_from_file(args.weight)

    # Initialize the scheduler module
    scheduler = Scheduler()

    # Run object detection + tracking
    with detection_graph.as_default():
        # Construct input and output Tensor names
        graph = tf.compat.v1.get_default_graph()
        tensor_frame = graph.get_tensor_by_name('image_tensor:0')
        tensor_nprop = graph.get_tensor_by_name('ApproxDet_num_proposals:0')
        tensor_shape = graph.get_tensor_by_name('ApproxDet_min_dim:0')
        output_tensor_dict = {}
        for key in ['num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes']:
            output_tensor_dict[key] = graph.get_tensor_by_name(key + ':0')

        with tf.compat.v1.Session(config=config) as sess:
            if args.preheat:
                preheat(sess, args.dataset_prefix, tensor_frame, tensor_nprop,
                        tensor_shape, output_tensor_dict)

            last_cl, last_ml, last_gl, last_contention_app = -1, -1, -1, ""
            for config in tqdm.tqdm(configs):
                # 0. Experiment control
                video_frame_path = config[0]
                req_type, req_value = config[1], float(config[2])

                # Update scheduler requirement
                assert req_type == "latency", "False user requirement type"
                scheduler.update(update_dict={"user_requirement": {req_type: req_value}})

                # Update contention environment 
                # note: gt_cpu_level, gt_mem_level, gt_gpu_level are black-box
                if len(config) == 6:
                    gt_cl, gt_ml, gt_gl = int(config[3]), int(config[4]), int(config[5])
                    if gt_cl != last_cl or gt_ml != last_ml or gt_gl != last_gl:
                        last_cl, last_ml, last_gl = gt_cl, gt_ml, gt_gl
                        with open("contention_level.txt", "w") as fout:
                            print("{} {} {}".format(gt_cl, gt_ml, gt_gl), file = fout)
                        time.sleep(15)
                elif len(config) == 4:
                     contention_app = config[3]
                     if contention_app != last_contention_app:
                        last_contention_app = contention_app
                        with open("contention_level.txt", "w") as fout:
                            print("{}".format(contention_app), file = fout)
                        time.sleep(15)

                # 1. Load a frame from the storage
                time1 = time.time()
                full_path = os.path.join(args.dataset_prefix, video_frame_path)
                image_pil = Image.open(full_path)
                time2 = time.time()

                # 2. Scheduler picks the right AB
                # Analyze video_frame_path, i.e., /data1/.../000000.JPEG
                vid_name = full_path.split("/")[-2], 
                frame_idx = int(full_path.split("/")[-1].split(".")[0])
                is_det_frame, is_scheduler_frame, is_new_video = \
                  scheduler.is_det_frame(vid_name, frame_idx)
                if is_scheduler_frame:
                    contention_levels = scheduler.sense_contention()
                    width, height = image_pil.size
                    sch_output = scheduler.schedule(contention_levels, height, width)
                else:
                    sch_output = scheduler.get_last_config()
                config, if_switch_tracker, est_acc, est_lat, feature_nobj, \
                  feature_avgsize = sch_output
                si, shape, nprop, tracker_name, ds = config

                if if_switch_tracker:
                    if tracker_name == "bboxmedianfixed":
                        tracker = FlowRawTracker(ds = ds, anchor = "fixed",
                                                 mode = "bbox_median")
                    else:
                        tracker = OpenCVTracker(ds = ds, name = tracker_name)
                time3 = time.time()

                # 3. detection framework starts to take over
                if is_det_frame:  # Do "detection"
                    # a1. Format change for detection DNN
                    image_np = np.array(image_pil).astype(np.uint8)
                    image_4D = np.expand_dims(image_np, axis=0)

                    # a2. Object detection DNN
                    feed_dict = {tensor_frame: image_4D, tensor_nprop: nprop,
                                 tensor_shape: shape}
                    output_dict = sess.run(output_tensor_dict, feed_dict=feed_dict)

                    # a3. Post processing
                    bboxes = output_dict_to_bboxes_single_img(output_dict)
                    time4 = time.time()

                    # b1. Object tracker routine
                    # tracker requires 3D numpy array in BGR
                    image_np = np.array(image_pil.convert('RGB'))[:, :, ::-1]
                    if isinstance(tracker, FlowRawTracker):
                        tracker.set_prev_frame(frame=image_np, bboxes=bboxes)
                    else:
                        tracker.reset_self()
                        tracker.set_prev_frame(frame=image_np, bboxes=bboxes)
                    time5 = time.time()
                else:  # Do "tracking"
                    time4 = time.time()

                    # b1. Format change for the tracker: an BGR numpy array
                    image_np = np.array(image_pil.convert('RGB'))[:, :, ::-1]

                    # b2. Object tracker
                    bboxes = tracker.track(image_np)
                    time5 = time.time()

                # 4. Feedback updates to the scheduler
                height, width = image_np.shape[:2]
                sizes = [(ymax-ymin)*(xmax-xmin)*height*width \
                   for _, _, ymin, xmin, ymax, xmax in bboxes]
                avgsize = np.sqrt(np.sum(sizes)) if sizes else 0
                nobj = len(bboxes)

                update_dict = {"last_avgsize": avgsize, "last_nobj": nobj}
                if is_det_frame:  # on detection frame
                    update_dict["last_avgsize_ondet"] = avgsize
                    update_dict["last_nobj_ondet"] = nobj
                if is_new_video:
                    update_dict["if_new_video"] = True
                    update_dict["height"] = height
                    update_dict["width"] = width
                scheduler.update(update_dict)

                detection_lat, tracker_lat = (time4-time3)*1e3, (time5-time4)*1e3
                if is_det_frame:  # on detection frame
                    scheduler.update(update_dict={"latency_log_type": "detection",
                                                  "latency_log_value": detection_lat})
                    scheduler.update(update_dict={"latency_log_type": "tracking",
                                                  "latency_log_value": tracker_lat})
                else:             # on tracking frame
                    scheduler.update(update_dict={"latency_log_type": "tracking",
                                                  "latency_log_value": tracker_lat})
                time6 = time.time()

                # 5a. per-obj detection log
                for cls, conf, ymin, xmin, ymax, xmax in bboxes:
                    print("{} {} {} {} {} {} {}".format(video_frame_path, cls,
                      conf, ymin, xmin, ymax, xmax), file = fout_det)

                # 5b. per-frame log
                loading_lat = (time2-time1)*1e3
                overhead_lat = (time3-time2+time6-time5)*1e3
                line = "{} {} {} ".format(video_frame_path, height, width)
                line += "{} {} ".format(is_det_frame, is_scheduler_frame)
                line += "{} ".format(if_switch_tracker)
                line += "{} {} {} {} {} ".format(si, shape, nprop, tracker_name, ds)
                line += "{:.3f} {:.3f} ".format(loading_lat, overhead_lat)
                line += "{:.3f} {:.3f} ".format(detection_lat, tracker_lat)
                line += "latency {} ".format(scheduler.user_requirement["latency"])
                line += "{:.4f} {:.3f} ".format(est_acc, est_lat)
                line += "{} ".format(contention_levels["cpu_level"])
                line += "{} ".format(contention_levels["mem_bw_level"])
                line += "{} ".format(contention_levels["gpu_level"])
                line += "{} {} {}".format(feature_nobj, feature_avgsize, nobj)
                for _, _, ymin, xmin, ymax, xmax in bboxes:
                    size = (ymax-ymin)*(xmax-xmin)
                    line += " {:.6f}".format(size)
                print(line, file=fout_lat)

    fout_det.close()
    fout_lat.close()
