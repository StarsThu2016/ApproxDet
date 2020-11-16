'''
Evaluate the accuracy and latency of an object detection + tracking model on a
  dataset. Shape, nprop, si, tracker_ds must be specified from the command.

Usage:
python3 ApproxKernel.py --imagefiles=test/VID_testimg_00106000.txt \
  --repeat=1 --preheat=1 \
  --dataset_prefix=/home/nvidia/ILSVRC2015/ \
  --shape=576 --nprop=100 --si=8 --tracker_ds=medianflow_ds4 \
  --weight=models/ApproxDet.pb \
  --output=test/VID_tmp.txt
'''

import numpy as np
import argparse, os, time, tqdm, getpass
from PIL import Image
import tensorflow as tf
from utils_approxdet.detection_helper import load_graph_from_file
from utils_approxdet.detection_helper import output_dict_to_bboxes_single_img
from utils_approxdet.tracker import OpenCVTracker, FlowRawTracker

def tracker_ds_to_config(tracker_ds):

    codebook = {
        "medianflow_ds4": {"tracker": "opencv", "ds": 4, "opencv_type": "medianflow"},
        "medianflow_ds2": {"tracker": "opencv", "ds": 2, "opencv_type": "medianflow"},
        "medianflow_ds1": {"tracker": "opencv", "ds": 1, "opencv_type": "medianflow"},
        "kcf_ds4": {"tracker": "opencv", "ds": 4, "opencv_type": "kcf"},
        "csrt_ds4": {"tracker": "opencv", "ds": 4, "opencv_type": "csrt"},
        "bboxmedianfixed_ds4":  {"tracker": "raw", "ds": 4, "anchor": "fixed",
                                 "mode": "bbox_median"}
    }
    return codebook[tracker_ds]
    
if __name__== "__main__":

    # Argument parsing
    parser = argparse.ArgumentParser(description=('Evaluate the latency and '
      'accuracy of the detection model on a dataset.'))
    parser.add_argument('--imagefiles', dest='imagefiles', required=True,
      help='The path to the video frames.')
    parser.add_argument('--repeat', dest='repeat', type=int, default=1,
      help='How many times do we repeatly run on imagefiles.')
    parser.add_argument('--preheat', dest='preheat',
      help='Whether to preheat the branches in detection DNN.')
    parser.add_argument('--nprop', dest='nprop', type=int, required=True,
      help='Number of proposals in the RPN of the detection DNN.')
    parser.add_argument('--shape', dest='shape', type=int, required=True,
      help='The resized shape of the video frames. (smaller of height and width)')
    parser.add_argument('--si', dest='si', type=int, required=True, 
      help='Samping interval of the detection DNN.')
    parser.add_argument('--tracker_ds', dest='tracker_ds', required=True,
      help='The tracker type and downsampling ratio')
    parser.add_argument('--dataset_prefix', dest='dataset_prefix',
      help='The path to the dataset.')
    parser.add_argument('--weight', dest='weight',
      help='The path to the weight file.')
    parser.add_argument('--output', dest='output', required=True,
      help='The filename of the detection and latency output, suffix will add.')
    args = parser.parse_args()

    # Hard-code the network configs
    dataset_prefix = args.dataset_prefix
    weight = args.weight

    # Load the detection DNN
    detection_graph = load_graph_from_file(weight)

    # Load the list of the test video frames
    with open(args.imagefiles) as f:
        lines = f.readlines()
    if args.shape and args.nprop:
        test_img_configs = [(line.strip().split()[0], False) for line in lines]
    test_img_configs = test_img_configs * args.repeat

    # Hard-code the path to the preheating video frames
    if args.preheat:
        img_dir = os.path.join(dataset_prefix, ("Data/VID/train/"
          "ILSVRC2015_VID_train_0002/ILSVRC2015_train_00703000"))
        configs = [(args.shape, args.nprop) for run in range(3)]
        img_configs = [("{}/{:06d}.JPEG".format(img_dir, idx), True) \
              for idx, config in enumerate(configs)]
        test_img_configs = img_configs + test_img_configs

    # Output log files
    si_str = "" if args.si == 8 else "_si" + str(args.si)
    detoutput_filename = args.output.rsplit(".", 1)[0] + \
      "_nprop{}_shape{}_{}{}_det.txt".format(args.nprop, args.shape,
                                             args.tracker_ds, si_str)
    latoutput_filename = args.output.rsplit(".", 1)[0] + \
      "_nprop{}_shape{}_{}{}_lat.txt".format(args.nprop, args.shape,
                                             args.tracker_ds, si_str)
    fout_det = open(detoutput_filename, "w")
    fout_lat = open(latoutput_filename, "w")

    # Set TensorFlow config
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    # Run object detection + tracking
    shape, nprop = args.shape, args.nprop
    si, tracker_ds = args.si, args.tracker_ds
    tracker_config = tracker_ds_to_config(tracker_ds)
    last_video = ""
    with detection_graph.as_default():
        graph = tf.compat.v1.get_default_graph()
        tensor_frame = graph.get_tensor_by_name('image_tensor:0')
        tensor_nprop = graph.get_tensor_by_name('ApproxDet_num_proposals:0')
        tensor_shape = graph.get_tensor_by_name('ApproxDet_min_dim:0')
        output_tensor_dict = {}
        for key in ['num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes']:
            output_tensor_dict[key] = graph.get_tensor_by_name(key + ':0')
        with tf.compat.v1.Session(config=config) as sess:
            for image_path, preheat_flag in tqdm.tqdm(test_img_configs):
                # 1a. Load a frame from the storage
                time1 = time.time()
                full_path = os.path.join(dataset_prefix, image_path)
                image_pil = Image.open(full_path)

                # 1b. (opertional) Analyze image_path, example "/data1/.../000000.JPEG"
                image_path_split = image_path.split("/")
                vid_name = image_path.split("/")[-2]
                img_idx = int(image_path.split("/")[-1].split(".")[0])
                time2 = time.time()

                # 2a. Init the tracker, if a new video comes
                if vid_name != last_video:
                    if tracker_config["tracker"] == 'raw':
                        tracker = FlowRawTracker(anchor = tracker_config["anchor"],
                                                 ds = tracker_config["ds"], 
                                                 mode = tracker_config["mode"])
                    else:
                        tracker = OpenCVTracker(ds = tracker_config["ds"], 
                                                name = tracker_config["opencv_type"])

                # 2b. Do "detection" or "tracking" based on the index of the video frame
                
                if img_idx % args.si == 0:  # Do "detection"
                    image_np = np.array(image_pil).astype(np.uint8)
                    image_4D = np.expand_dims(image_np, axis=0)
                    feed_dict = {tensor_frame: image_4D, tensor_nprop: nprop,
                                 tensor_shape: shape}
                    output_dict = sess.run(output_tensor_dict, feed_dict = feed_dict)
                    bboxes = output_dict_to_bboxes_single_img(output_dict)

                    if isinstance(tracker, OpenCVTracker):
                        tracker.reset_self()
                    image_np = np.array(image_pil.convert('RGB'))[:, :, ::-1]
                    tracker.set_prev_frame(frame=image_np, bboxes=bboxes)
                else:                  # Do "tracking"
                    image_np = np.array(image_pil.convert('RGB'))[:, :, ::-1]
                    bboxes = tracker.track(image_np)
                last_video = vid_name
                time3 = time.time()

                # 4. Print the detection bounding boxes, latency results
                if not preheat_flag:
                    for cls, conf, ymin, xmin, ymax, xmax in bboxes:
                        print("{} {} {} {} {} {} {}".format(image_path, cls,
                          conf, ymin, xmin, ymax, xmax), file = fout_det)
                    nobj = len(bboxes)
                    height, width = image_np.shape[:2]
                    loading_lat, inf_lat = (time2-time1)*1e3, (time3-time2)*1e3
                    line = "{} {} {} {} {} {} {} {}".format(image_path, height,
                      width, shape, nprop, loading_lat, inf_lat, nobj)
                    for _, _, ymin, xmin, ymax, xmax in bboxes:
                        size = (ymax-ymin)*(xmax-xmin)
                        line += " {}".format(size)
                    print(line, file = fout_lat)
    fout_det.close()
    fout_lat.close()
