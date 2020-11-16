'''
Evaluate the accuracy and latency of an object detection + tracking model on a 
  dataset, speeding up by loading the detection results from a saved file 
  instead of running the detection DNN.

Usage,
python3 ApproxTracking.py --imagefiles=test/VID_testimg_00106000.txt \
  --detection_file=test/VID_valset_nprop100_shape576_det.txt \
  --dataset_prefix=/home/nvidia/ILSVRC2015/ \
  --repeat=1 --si=8 --tracker_ds=medianflow_ds4 \
  --output=test/VID_valset_nprop100_shape576.txt
'''

import numpy as np
import argparse, os, time, tqdm
from PIL import Image
from utils_approxdet.tracker import OpenCVTracker, FlowRawTracker
from utils_approxdet.tracker import import_detection_file

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

# main function
if __name__== "__main__":

    parser = argparse.ArgumentParser(description=('Evaluate the accuracy and '
      'latency of an object detection + tracking model on a dataset.'))
    parser.add_argument('--imagefiles', dest='imagefiles', required=True,
      help='The path to the video frames.')
    parser.add_argument('--detection_file', dest='detection_file', required=True,
      help='The saved detection results.')
    parser.add_argument('--repeat', dest='repeat', type=int, default=1,
      help='How many times do we repeatly run on imagefiles.')
    parser.add_argument('--si', dest='si', type=int, required=True,
      help='Samping interval of the detection DNN.')
    parser.add_argument('--tracker_ds', dest='tracker_ds',
      help='The tracker type and downsampling ratio')
    parser.add_argument('--dataset_prefix', dest='dataset_prefix',
      help='The path to the dataset.')
    parser.add_argument('--output', dest='output', required=True,
      help='Filename of the detection and latency output, suffix will add.')
    args = parser.parse_args()

    # Load list of test images
    with open(args.imagefiles) as f:
        lines = f.readlines()
    if args.tracker_ds:
        img_configs = [(x.strip().split()[0], args.tracker_ds) for x in lines]
    else:
        imgs = [line.strip().split()[0] for line in lines]
        tracker_dses = [line.strip().split()[1] for line in lines]
        img_configs = [(ti, ttd) for ti, ttd in zip(imgs, tracker_dses)]
    test_img_configs = img_configs * args.repeat

    # Output log files
    if args.tracker_ds:
        detoutput = "{}_si{}_{}_det.txt".format(args.output.rsplit(".", 1)[0],
                                                args.si, args.tracker_ds)
        latoutput = "{}_si{}_{}_lat.txt".format(args.output.rsplit(".", 1)[0],
                                                args.si, args.tracker_ds)
    else: 
        detoutput = "{}_si{}_det.txt".format(args.output.rsplit(".", 1)[0],
                                             args.si)
        latoutput = "{}_si{}_lat.txt".format(args.output.rsplit(".", 1)[0],
                                             args.si)

    # Detection maps "name_of_image" to "list of bboxes"
    #  -- (cls, conf, ymin, xmin, ymax, xmax)
    detection = import_detection_file(args.detection_file)
    
    # Run object tracking
    last_video, last_tracker_ds = "", ""
    fout_det = open(detoutput, "w")
    fout_lat = open(latoutput, "w")

    for image_path, tracker_ds in tqdm.tqdm(test_img_configs):
        # 1a. Load a frame from the storage
        time1 = time.time()
        full_path = os.path.join(args.dataset_prefix, image_path)
        image_pil = Image.open(full_path)
        time2 = time.time()

        # 1b. (opertional) Analyze image_path, example "/data1/.../000000.JPEG"
        image_path_split = image_path.split("/")
        vid_name = image_path.split("/")[-2]
        img_idx = int(image_path.split("/")[-1].split(".")[0])
        tracker_config = tracker_ds_to_config(tracker_ds)

        # 2a. Init the tracker, if a new video comes or a new tracker_ds
        if vid_name != last_video or tracker_ds != last_tracker_ds:
            if tracker_config["tracker"] == 'raw':
                tracker = FlowRawTracker(anchor = tracker_config["anchor"],
                                         ds = tracker_config["ds"],
                                         mode = tracker_config["mode"])
            else:
                tracker = OpenCVTracker(ds = tracker_config["ds"],
                                        name = tracker_config["opencv_type"])
            last_tracker_ds = tracker_ds
            last_video = vid_name

        # 2b. Do "detection" or "tracking" based on the index of the video frame
        image_np = np.array(image_pil.convert('RGB'))[:, :, ::-1]
        if img_idx % args.si == 0:  # Do "detection"
            bboxes = detection[image_path]
            if isinstance(tracker, OpenCVTracker):
                tracker.reset_self()
            tracker.set_prev_frame(frame=image_np, bboxes=bboxes)
        else:                  # Do "tracking"
            bboxes = tracker.track(image_np)
        time3 = time.time()

        # 3. Print the detection bounding boxes, latency results to output files
        for cls, conf, ymin, xmin, ymax, xmax in bboxes:
            print("{} {} {} {} {} {} {}".format(image_path, cls, conf,
              ymin, xmin, ymax, xmax), file = fout_det)

        height, width = image_np.shape[:2]
        loading_lat, inference_lat = (time2-time1)*1e3, (time3-time2)*1e3
        nobj = len(bboxes)
        line = "{} {} {} {} {} {} {}".format(image_path, height, width,
          tracker_ds, loading_lat, inference_lat, nobj)
        for _, _, ymin, xmin, ymax, xmax in bboxes:
            size = (ymax-ymin)*(xmax-xmin)
            line += " {}".format(size)
        print(line, file = fout_lat)

    fout_det.close()
    fout_lat.close()
