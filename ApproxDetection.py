'''
Evaluate the accuracy and latency of an object detection model on a dataset.

Usage:
python3 ApproxDetection.py --imagefiles=test/VID_testimg_00106000.txt \
  --repeat=1 --preheat=1 --dataset_prefix=/home/nvidia/ILSVRC2015/ \
  --shape=576 --nprop=100 --weight=models/ApproxDet.pb \
  --output=test/VID_tmp.txt
'''

import numpy as np
import argparse, os, time, tqdm
from PIL import Image
import tensorflow as tf
from utils_approxdet.detection_helper import load_graph_from_file
from utils_approxdet.detection_helper import output_dict_to_bboxes_single_img

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
    parser.add_argument('--nprop', dest='nprop', type=int, 
      help='Number of proposals in the RPN of the detection DNN.')
    parser.add_argument('--shape', dest='shape', type=int, 
      help='The resized shape of the video frames. (smaller of height and width)')
    parser.add_argument('--dataset_prefix', dest='dataset_prefix',
      help='The path to the dataset.')
    parser.add_argument('--weight', dest='weight',
      help='The path to the weight file.')
    parser.add_argument('--output', dest='output', required=True,
      help='The filename of the detection and latency output, suffix will add.')
    args = parser.parse_args()

    # Hard-code the network configs
    shapes = [224, 320, 448, 576]
    nprops = [100, 50, 20, 10, 5, 3, 1]
    dataset_prefix = args.dataset_prefix
    weight = args.weight

    # Load the detection DNN
    detection_graph = load_graph_from_file(weight)

    # Load the list of the test video frames
    with open(args.imagefiles) as f:
        lines = f.readlines()
    if args.shape and args.nprop:
        test_img_configs = [(line.strip().split()[0], args.shape, args.nprop,
                             False) for line in lines]
    else:
        test_imgs = [line.strip().split()[0] for line in lines]
        test_shapes = [int(line.strip().split()[1]) for line in lines]
        test_nprops = [int(line.strip().split()[2]) for line in lines]
        test_img_configs = [(ti, ts, tn, False) for ti, ts, tn in \
                            zip(test_imgs, test_shapes, test_nprops)]
    test_img_configs = test_img_configs * args.repeat

    # Hard-code the path to the preheating video frames
    if args.preheat:
        img_dir = "Data/VID/train/ILSVRC2015_VID_train_0002/ILSVRC2015_train_00703000"
        if not (args.shape and args.nprop):
            # 28 configs * 3 images per config
            configs = [(shape, nprop) for shape in shapes for nprop in nprops \
                                      for run in range(3)]      
        else:  # For baseline, args.shape and args.nprop must be specified
            # 1 config * 3 images 
            configs = [(args.shape, args.nprop) for run in range(3)]
        img_configs = [("{}/{:06d}.JPEG".format(img_dir, idx), *config, True) \
              for idx, config in enumerate(configs)]
        test_img_configs = img_configs + test_img_configs

    # Output log files
    if args.shape and args.nprop:
        detoutput_filename = args.output.rsplit(".", 1)[0] + \
          "_nprop{}_shape{}_det.txt".format(args.nprop, args.shape)
        latoutput_filename = args.output.rsplit(".", 1)[0] + \
          "_nprop{}_shape{}_lat.txt".format(args.nprop, args.shape)
    else:
        detoutput_filename = args.output.rsplit(".", 1)[0] + "_det.txt".format()
        latoutput_filename = args.output.rsplit(".", 1)[0] + "_lat.txt".format()
    fout_det = open(detoutput_filename, "w")
    fout_lat = open(latoutput_filename, "w")

    # Set TensorFlow config
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    # Run object detection
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
            for path, shape, nprop, preheat_flag in tqdm.tqdm(test_img_configs):
                # 1. Load a frame from the storage
                time1 = time.time()
                full_path = os.path.join(dataset_prefix, path)
                image_pil = Image.open(full_path)

                # 2. Run the detection DNN
                time2 = time.time()
                image_np = np.array(image_pil).astype(np.uint8)
                image_4D = np.expand_dims(image_np, axis=0)
                feed_dict = {tensor_frame: image_4D, tensor_nprop: nprop,
                             tensor_shape: shape}
                output_dict = sess.run(output_tensor_dict, feed_dict = feed_dict)
                time3 = time.time()

                # 3. Post processing
                bboxes = output_dict_to_bboxes_single_img(output_dict)

                # 4. Print the detection bounding boxes, latency results
                if not preheat_flag:
                    for cls, conf, ymin, xmin, ymax, xmax in bboxes:
                        print("{} {} {} {} {} {} {}".format(path, cls,
                          conf, ymin, xmin, ymax, xmax), file = fout_det)
                    nobj = len(bboxes)
                    height, width = image_np.shape[:2]
                    loading_lat, inf_lat = (time2-time1)*1e3, (time3-time2)*1e3
                    line = "{} {} {} {} {} {} {} {}".format(path, height,
                      width, shape, nprop, loading_lat, inf_lat, nobj)
                    for _, _, ymin, xmin, ymax, xmax in bboxes:
                        size = (ymax-ymin)*(xmax-xmin)
                        line += " {}".format(size)
                    print(line, file = fout_lat)
    fout_det.close()
    fout_lat.close()
