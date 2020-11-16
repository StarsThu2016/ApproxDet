'''
Evaluate the switching overhead of an object detection model along the "shape"
  and "number of proposals" tuning knobs, on a 800+ frame video snippet.
All 28*27=756 pairs are studied.
  shape(4) in [224, 320, 448, 576]
  nprop(7) in [1, 3, 5, 10, 20, 50, 100]
Preheating and test images are both from a validation snippet 
  "Data/VID/train/ILSVRC2015_VID_train_0002/ILSVRC2015_train_00703000"

Usage:
python3 ApproxDetectionProfilerSw.py --dataset_prefix=/home/nvidia/ILSVRC2015/ \
  --weight=models/ApproxDet.pb --output=test/VID_switchingoverhead_run0.txt
'''

import numpy as np
import argparse, os, time, random, tqdm
from PIL import Image
import tensorflow as tf
from utils_approxdet.detection_helper import load_graph_from_file
from utils_approxdet.detection_helper import output_dict_to_bboxes_single_img

if __name__== "__main__":

    # Argument parsing
    parser = argparse.ArgumentParser(description=('Evaluate the switching '
      'overhead of an object detection model along the "shape" and "number of '
      'proposals" tuning knobs'))
    parser.add_argument('--dataset_prefix', dest='dataset_prefix',
      help='The path to the dataset.')
    parser.add_argument('--weight', dest='weight',
      help='The path to the weight file.')
    parser.add_argument('--output', dest='output', required=True, 
      help='The filename of the latency logs.')
    args = parser.parse_args()

    # Output log file
    fout = open(args.output, "w")

    # Set TensorFlow config
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    # Load detection DNN
    detection_graph = load_graph_from_file(args.weight)
            
    # Load the list of test video frames: 
    #   28 branches * 3 images + 28*27 switching pairs * 20 images = 15204
    img_dir = "Data/VID/train/ILSVRC2015_VID_train_0002/ILSVRC2015_train_00703000"
    image_paths = ["{}/{:06d}.JPEG".format(img_dir, x%16+800) for x in range(28*3)] + \
                  ["{}/{:06d}.JPEG".format(img_dir, x%800) for x in range(28*27*20)]

    # Construct the experiments
    all_shapes = [224, 320, 448, 576]
    all_nprops = [100, 50, 20, 10, 5, 3, 1]
    configs = [(nprop, shape) for nprop in all_nprops for shape in all_shapes]
    all_pairs = []
    for from_branch in range(28):
        for to_branch in range(28):
            if from_branch != to_branch:
                all_pairs.append((configs[from_branch][0], configs[from_branch][1],
                                  configs[to_branch][0], configs[to_branch][1]))
    random.shuffle(all_pairs)

    nprops, shapes = [], []
    for nprop, shape in configs:
        nprops += [nprop] * 3
        shapes += [shape] * 3
    for from_nprop, from_shape, to_nprop, to_shape in all_pairs:
        nprops += [from_nprop] * 10
        nprops += [to_nprop] * 10
        shapes += [from_shape] * 10
        shapes += [to_shape] * 10

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
            for idx in tqdm.tqdm(range(len(image_paths))): 
                path, nprop, shape = image_paths[idx], nprops[idx], shapes[idx]
                # 1. Load a frame from storage
                time1 = time.time()
                full_path = os.path.join(args.dataset_prefix, path)
                image_pil = Image.open(full_path)

                # 2. detection DNN
                time2 = time.time()
                image_np = np.array(image_pil).astype(np.uint8)
                image_4D = np.expand_dims(image_np, axis=0) 
                feed_dict = {tensor_frame: image_4D, tensor_nprop: nprop,
                             tensor_shape: shape}
                output_dict = sess.run(output_tensor_dict, feed_dict = feed_dict)
                time3 = time.time()

                # 3. Post processing
                bboxes = output_dict_to_bboxes_single_img(output_dict)

                # 4. Print latency results to file
                if idx >= 28*3: # after pre-heating
                    nobj = len(bboxes)
                    height, width = image_np.shape[:2]
                    loading_lat, inf_lat = (time2-time1)*1e3, (time3-time2)*1e3
                    sizes = [(ymax-ymin)*(xmax-xmin)*height*width \
                             for _, _, ymin, xmin, ymax, xmax in bboxes]
                    avgsize = np.sqrt(np.sum(sizes)) if sizes else 0
                    print("{} {} {} {} {} {} {}".format(path, nprop, shape,
                      nobj, avgsize, loading_lat, inf_lat), file = fout)
    fout.close()

