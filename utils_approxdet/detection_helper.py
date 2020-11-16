import numpy as np
import tensorflow as tf

def load_graph_from_file(filename):

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(filename, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def output_dict_to_bboxes_single_img(output_dict):

    # Output translation, in (cls, conf, ymin, xmin, ymax, xmax) in [0,1] range
    # all outputs are float32 numpy arrays, so convert types as appropriate
    N = int(output_dict['num_detections'][0])
    boxes = [(cls-1, sc, box[0], box[1], box[2], box[3]) for cls, box, sc in \
      zip(output_dict['detection_classes'][0].astype(np.int64)[:N],
          output_dict['detection_boxes'][0][:N],
          output_dict['detection_scores'][0][:N])]
    return boxes
